#!/usr/bin/env node
import { spawn } from 'child_process';
import * as readline from 'readline';

export const WORKSPACE = `${process.env.HOME}/.agent0`;
export const MODEL = "minimax-m2.5:cloud";
export const MAX_TURNS = 5;

const conversationHistory: string[] = [];
const keyInfo: string[] = [];

async function callOllama(prompt: string, system: string = ""): Promise<string> {
  const fullPrompt = system ? `${system}\n\n${prompt}` : prompt;

  const payload = {
    model: MODEL,
    prompt: fullPrompt,
    stream: false
  };

  try {
    const resp = await fetch("http://localhost:11434/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await resp.json();
    return (result.response as string).trim();
  } catch (e) {
    return `Error: ${e}`;
  }
}

function buildContext(): string {
  const contextParts: string[] = [];

  if (keyInfo.length > 0) {
    const itemsXml = keyInfo.map(k => `  <item>${k}</item>`).join("\n");
    contextParts.push(`<memory>\n${itemsXml}\n</memory>`);
  }

  if (conversationHistory.length > 0) {
    const recent = conversationHistory.slice(-MAX_TURNS * 2);
    contextParts.push("<history>\n" + recent.join("\n") + "\n</history>");
  }

  return contextParts.join("\n\n");
}

function updateMemory(userInput: string, assistantResponse: string, toolResult?: string): void {
  conversationHistory.push(`  <user>${userInput}</user>`);
  conversationHistory.push(`  <assistant>${assistantResponse}</assistant>`);
  if (toolResult) {
    conversationHistory.push(`  <tool>${toolResult.slice(0, 500)}</tool>`);
  }

  while (conversationHistory.length > MAX_TURNS * 4) {
    conversationHistory.shift();
  }
}

async function extractKeyInfo(userInput: string, assistantResponse: string): Promise<void> {
  const extractPrompt = `根據這段對話，有沒有需要長期記憶的關鍵資訊？
如果有，用以下格式輸出（最多 2 項）。如果沒有，輸出 <memory></memory>。

<memory>
  <item>要記憶的資訊 1</item>
  <item>要記憶的資訊 2</item>
</memory>

對話：
<user>${userInput}</user>
<assistant>${assistantResponse}</assistant>`;

  try {
    const result = await callOllama(extractPrompt, "");
    const matches = result.match(/<item>(.*?)<\/item>/gs);
    if (matches) {
      for (const match of matches) {
        const item = match.replace(/<\/?item>/g, "").trim();
        if (item && !keyInfo.includes(item)) {
          keyInfo.push(item);
        }
      }
    }
  } catch (e) {
    // silently fail
  }
}

export const SYSTEM_PROMPT = `你是 Jarvis，一個有用的 AI 助理。

重要規則：
1. 當你需要執行 shell 命令時，必須用 <shell> 標籤包住命令
2. <shell> 標籤內可以是多行命令（用反斜槓 \\ 或 && 連接）
3. 當你完成所有操作後，用 <end/> 結束你的回覆

流程：
- 如果需要執行命令，輸出 <shell>...</shell>
- 執行完後我會顯示結果
- 如果還需要更多命令，繼續輸出 <shell>
- 當完成所有操作後，輸出 <end/> 表示結束`;

async function main() {
  const fs = await import('fs');
  if (!fs.existsSync(WORKSPACE)) {
    fs.mkdirSync(WORKSPACE, { recursive: true });
  }

  console.log(`Agent0 - ${MODEL}（含記憶功能）`);
  console.log(`工作區：${WORKSPACE}`);
  console.log("指令：/quit、/memory（顯示關鍵資訊）\n");

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  const askUser = (): Promise<string> => new Promise(resolve => {
    rl.question("你：", (answer: string) => resolve(answer.trim()));
  });

  while (true) {
    let userInput: string;
    try {
      userInput = await askUser();
    } catch (e) {
      console.log("\n再見！");
      break;
    }

    if (!userInput) continue;
    if (["quit", "exit", "/q"].includes(userInput.toLowerCase())) {
      console.log("再見！");
      break;
    }
    if (userInput.toLowerCase() === "/memory") {
      console.log(`關鍵資訊：${JSON.stringify(keyInfo)}`);
      continue;
    }

    const context = buildContext();
    const fullPrompt = context
      ? `${context}\n\n<user>${userInput}</user>`
      : `<user>${userInput}</user>`;

    let response = await callOllama(fullPrompt, SYSTEM_PROMPT);

    let toolResult: string | undefined;
    let currentResponse = response;

    while (true) {
      if (currentResponse.includes("<end/>")) {
        response = currentResponse.split("<end/>")[0].trim();
        break;
      }

      const shellMatches = currentResponse.match(/<shell>([\s\S]*?)<\/shell>/g);
      if (!shellMatches) {
        response = currentResponse;
        break;
      }

      const allOutputs: string[] = [];

      for (const match of shellMatches) {
        const cmd = match.replace(/<\/?shell>/g, "").trim();
        try {
          const result = await new Promise<{ stdout: string; stderr: string }>((resolve, reject) => {
            const proc = spawn(cmd, { shell: true, cwd: process.cwd() });
            let stdout = "";
            let stderr = "";
            proc.stdout?.on("data", (d: Buffer) => stdout += d.toString());
            proc.stderr?.on("data", (d: Buffer) => stderr += d.toString());
            proc.on("close", (code: number | null) => resolve({ stdout, stderr }));
            proc.on("error", reject);
            setTimeout(() => {
              proc.kill();
              reject(new Error("Timeout"));
            }, 30000);
          });
          const output = result.stdout + result.stderr;
          console.log(`\n=== 執行命令 ===\n${cmd}\n\n結果：${output || "（無輸出）"}\n`);
          allOutputs.push(`$ ${cmd}\n${output || "（無輸出）"}`);
        } catch (e: unknown) {
          const errMsg = e instanceof Error ? e.message : String(e);
          console.log(`錯誤：${errMsg}`);
          allOutputs.push(`$ ${cmd}\n錯誤：${errMsg}`);
        }
      }

      toolResult = (toolResult || "") + "\n" + allOutputs.join("\n");

      const followUpPrompt = `<context>${context}</context>

<user>${userInput}</user>
<assistant>${currentResponse}</assistant>
<output>
${allOutputs.join("\n")}
</output>

如果需要更多命令就輸出 <shell>。否則，輸出 <end/> 表示結束：`;

      currentResponse = await callOllama(followUpPrompt, SYSTEM_PROMPT);
    }

    console.log(`\n🤖 ${response}\n`);

    updateMemory(userInput, response, toolResult);
    if (toolResult) {
      await extractKeyInfo(userInput, response);
    }
  }

  rl.close();
}

if (process.argv[1] && import.meta.url.includes(process.argv[1])) {
  main().catch(console.error);
}