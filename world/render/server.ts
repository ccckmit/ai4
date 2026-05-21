import http from 'node:http';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { execSync } from 'node:child_process';
import { WebSocketServer, WebSocket } from 'ws';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let httpServer: http.Server | null = null;
let wss: WebSocketServer | null = null;
const clients = new Set<WebSocket>();

export interface CartPoleFrame {
  x: number;
  theta: number;
  steps: number;
  reward: number;
  done: boolean;
}

function loadHTML(): string {
  const htmlPath = join(__dirname, 'viewer.html');
  if (existsSync(htmlPath)) {
    return readFileSync(htmlPath, 'utf-8');
  }
  return '<html><body><h1>viewer.html not found</h1></body></html>';
}

function openBrowser(url: string): void {
  const cmd = process.platform === 'darwin' ? 'open' :
              process.platform === 'win32' ? 'start' : 'xdg-open';
  try {
    execSync(`${cmd} ${url}`, { stdio: 'ignore', timeout: 3000 });
  } catch { }
}

export function startServer(port = 8080): void {
  if (httpServer) return;

  const html = loadHTML();

  httpServer = http.createServer((req, res) => {
    if (req.url === '/') {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(html);
    } else {
      res.writeHead(404);
      res.end('Not Found');
    }
  });

  wss = new WebSocketServer({ server: httpServer });

  wss.on('connection', (ws) => {
    clients.add(ws);
    ws.on('close', () => clients.delete(ws));
    ws.on('error', () => clients.delete(ws));
  });

  httpServer.listen(port, () => {
    console.log(`[render] http://localhost:${port}  (open in browser)`);
  });

  // small delay so server is ready before browser opens
  setTimeout(() => openBrowser(`http://localhost:${port}`), 300);
}

export function sendFrame(data: CartPoleFrame): void {
  if (!httpServer) startServer();
  if (clients.size === 0) return;

  const msg = JSON.stringify(data);
  for (const ws of clients) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(msg);
    }
  }
}

export function stopServer(): void {
  for (const ws of clients) {
    ws.close();
    clients.delete(ws);
  }
  wss?.close();
  httpServer?.close();
  httpServer = null;
  wss = null;
}
