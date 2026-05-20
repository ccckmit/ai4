import * as agent from '../agent';

function assert(condition: boolean, msg: string) {
  if (!condition) throw new Error(`Assertion failed: ${msg}`);
}

function testConstants() {
  assert(agent.WORKSPACE.endsWith(".agent0"), `WORKSPACE should end with .agent0, got ${agent.WORKSPACE}`);
  assert(agent.MODEL === "minimax-m2.5:cloud", `MODEL should be minimax-m2.5:cloud, got ${agent.MODEL}`);
  assert(agent.MAX_TURNS === 5, `MAX_TURNS should be 5, got ${agent.MAX_TURNS}`);
  console.log('  [PASS] Constants');
}

function testSystemPrompt() {
  assert(agent.SYSTEM_PROMPT.length > 0, "SYSTEM_PROMPT should not be empty");
  assert(agent.SYSTEM_PROMPT.includes("<shell>"), "SYSTEM_PROMPT should contain <shell>");
  assert(agent.SYSTEM_PROMPT.includes("<end/>"), "SYSTEM_PROMPT should contain <end/>");
  console.log('  [PASS] SystemPrompt');
}

function testShellTagParsing() {
  const response1 = "Hello <shell>echo test</shell> world";
  const matches1 = response1.match(/<shell>([\s\S]*?)<\/shell>/g);
  assert(matches1 !== null && matches1.length === 1, `Expected 1 shell match, got ${matches1?.length}`);
  assert(matches1![0].replace(/<\/?shell>/g, "").trim() === "echo test", "Shell content should be 'echo test'");

  const response2 = "<shell>ls</shell> then <shell>cat file.txt</shell>";
  const matches2 = response2.match(/<shell>([\s\S]*?)<\/shell>/g);
  assert(matches2 !== null && matches2.length === 2, `Expected 2 shell matches, got ${matches2?.length}`);

  const response3 = "Just text";
  const matches3 = response3.match(/<shell>([\s\S]*?)<\/shell>/g);
  assert(matches3 === null || matches3.length === 0, "Should have no shell matches");

  console.log('  [PASS] ShellTagParsing');
}

function testEndTag() {
  const response1 = "Some text <end/> more";
  assert(response1.includes("<end/>"), "Should detect <end/> tag");

  const response2 = "No end tag here";
  assert(!response2.includes("<end/>"), "Should not detect <end/> tag");

  console.log('  [PASS] EndTag');
}

function testExtractKeyInfoRegex() {
  const result = `<memory>
  <item>記住用戶喜歡貓</item>
  <item>用戶住在台北</item>
</memory>`;
  const matches = result.match(/<item>(.*?)<\/item>/gs);
  assert(matches !== null && matches.length === 2, `Expected 2 matches, got ${matches?.length}`);

  const result2 = "<memory></memory>";
  const matches2 = result2.match(/<item>(.*?)<\/item>/gs);
  assert(matches2 === null || matches2.length === 0, "Should have no matches for empty memory");

  console.log('  [PASS] ExtractKeyInfoRegex');
}

console.log('\n=== llm agent Tests ===');
testConstants();
testSystemPrompt();
testShellTagParsing();
testEndTag();
testExtractKeyInfoRegex();
console.log('\n  all passed');