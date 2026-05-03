// AI Proxy Remote Chat — drive VS Code's chat panel via HTTP.
// Vanilla JS, no build step. Single file; just sideload via "Install from Location".

const vscode = require('vscode');
const http = require('http');
const https = require('https');
const os = require('os');
const url = require('url');

let server = null;
let statusBar = null;
let registerTimer = null;

function cfg() {
  const c = vscode.workspace.getConfiguration('aiProxyRemoteChat');
  return {
    port: c.get('port', 13337),
    bind: c.get('bind', '127.0.0.1'),
    token: c.get('token', ''),
    defaultCommand: c.get('defaultCommand', 'workbench.action.chat.open'),
    newChatCommand: c.get('newChatCommand', 'workbench.action.chat.newChat'),
    submitCommand: c.get('submitCommand', 'workbench.action.chat.submit'),
    alwaysSubmit: c.get('alwaysSubmit', true),
    proxyUrl: c.get('proxyUrl', '').trim(),
    advertiseUrl: c.get('advertiseUrl', '').trim(),
    advertiseName: c.get('advertiseName', '').trim(),
  };
}

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

function _firstLanIP() {
  // Pick a non-internal IPv4 address to advertise to the proxy when advertiseUrl is unset.
  const ifs = os.networkInterfaces();
  for (const list of Object.values(ifs)) {
    for (const i of list || []) {
      if (i.family === 'IPv4' && !i.internal) return i.address;
    }
  }
  return '127.0.0.1';
}

function _post(targetUrl, body, headers) {
  return new Promise((resolve, reject) => {
    const u = url.parse(targetUrl);
    const lib = u.protocol === 'https:' ? https : http;
    const data = Buffer.from(JSON.stringify(body), 'utf8');
    const req = lib.request({
      method: 'POST',
      protocol: u.protocol, hostname: u.hostname, port: u.port,
      path: u.path,
      headers: Object.assign({
        'content-type': 'application/json',
        'content-length': data.length,
      }, headers || {}),
      timeout: 5000,
    }, (res) => {
      const chunks = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => resolve({ status: res.statusCode, text: Buffer.concat(chunks).toString('utf8') }));
    });
    req.on('error', reject);
    req.on('timeout', () => { req.destroy(new Error('timeout')); });
    req.write(data); req.end();
  });
}

async function selfRegister() {
  const c = cfg();
  if (!c.proxyUrl) return;
  const advertise = c.advertiseUrl || `http://${_firstLanIP()}:${c.port}`;
  const name = c.advertiseName || os.hostname();
  try {
    const r = await _post(c.proxyUrl.replace(/\/+$/, '') + '/__proxy/api/control/register', {
      name, url: advertise, token: c.token || null, kind: 'vscode-chat',
    });
    if (r.status >= 200 && r.status < 300) {
      console.log(`AI Proxy Remote Chat: registered "${name}" -> ${advertise} with ${c.proxyUrl}`);
    } else {
      console.warn(`AI Proxy Remote Chat: registration failed (HTTP ${r.status}): ${r.text.slice(0, 200)}`);
    }
  } catch (e) {
    console.warn(`AI Proxy Remote Chat: registration error: ${e.message}`);
  }
}

function ok(res, payload) {
  res.writeHead(200, { 'content-type': 'application/json' });
  res.end(JSON.stringify(payload || { ok: true }));
}

function err(res, code, msg) {
  res.writeHead(code, { 'content-type': 'application/json' });
  res.end(JSON.stringify({ error: msg }));
}

async function readJson(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', c => chunks.push(c));
    req.on('end', () => {
      const text = Buffer.concat(chunks).toString('utf8') || '{}';
      try { resolve(JSON.parse(text)); }
      catch (e) { reject(new Error('invalid JSON: ' + e.message)); }
    });
    req.on('error', reject);
  });
}

async function handle(req, res) {
  const c = cfg();
  // CORS for casual curl/browser hits.
  res.setHeader('access-control-allow-origin', '*');
  res.setHeader('access-control-allow-headers', 'authorization, content-type');
  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  // Auth (skip if no token configured AND bound to localhost).
  if (c.token) {
    const auth = req.headers['authorization'] || '';
    if (auth !== `Bearer ${c.token}`) return err(res, 401, 'unauthorized');
  } else if (c.bind !== '127.0.0.1' && c.bind !== 'localhost') {
    return err(res, 403, 'no token configured but binding non-loopback — refusing');
  }

  if (req.method === 'GET' && req.url === '/status') {
    return ok(res, {
      service: 'ai-proxy-remote-chat',
      bind: c.bind, port: c.port,
      defaultCommand: c.defaultCommand,
      vscode: vscode.version,
    });
  }

  if (req.method === 'POST' && (req.url === '/chat' || req.url === '/prompt')) {
    let body;
    try { body = await readJson(req); } catch (e) { return err(res, 400, e.message); }
    const prompt = body.prompt || body.query;
    if (typeof prompt !== 'string' || !prompt.trim()) return err(res, 400, "missing 'prompt'");
    const command = body.command || c.defaultCommand;
    const newChatCmd = body.newChatCommand || c.newChatCommand;
    const submitCmd = body.submitCommand || c.submitCommand;
    const wantsNewChat = body.newChat === true;
    const wantsSubmit = body.submit !== undefined ? !!body.submit : c.alwaysSubmit;
    const args = { query: prompt };
    if (body.location) args.location = body.location;          // 'panel' | 'editor' | 'terminal'
    if (body.isPartialQuery !== undefined) args.isPartialQuery = !!body.isPartialQuery;
    if (body.attachScreenshot) args.attachScreenshot = true;
    const trace = [];
    try {
      if (wantsNewChat && newChatCmd) {
        trace.push(`new-chat:${newChatCmd}`);
        await vscode.commands.executeCommand(newChatCmd);
        await sleep(120);  // give the new chat panel a moment to focus the input
      }
      trace.push(`open:${command}`);
      await vscode.commands.executeCommand(command, args);
      if (wantsSubmit && submitCmd) {
        await sleep(80);
        try {
          await vscode.commands.executeCommand(submitCmd);
          trace.push(`submit:${submitCmd}`);
        } catch (e) {
          // Submit command may not exist for some chat providers; not fatal — `query` arg
          // alone usually triggers submission for built-in chat.
          trace.push(`submit-skipped:${String(e).slice(0, 60)}`);
        }
      }
      return ok(res, { ok: true, length: prompt.length, trace });
    } catch (e) {
      return err(res, 500, `vscode command failed: ${String(e)} (trace: ${trace.join(' → ')})`);
    }
  }

  if (req.method === 'POST' && req.url === '/command') {
    // Escape hatch: run any VS Code command. Useful for debugging / power users.
    let body;
    try { body = await readJson(req); } catch (e) { return err(res, 400, e.message); }
    if (!body.id) return err(res, 400, "missing 'id'");
    try {
      const result = await vscode.commands.executeCommand(body.id, ...(body.args || []));
      return ok(res, { ok: true, result: result === undefined ? null : result });
    } catch (e) {
      return err(res, 500, `vscode command failed: ${String(e)}`);
    }
  }

  err(res, 404, 'not found');
}

function startServer() {
  if (server) { try { server.close(); } catch {} server = null; }
  const c = cfg();
  server = http.createServer((req, res) => {
    handle(req, res).catch(e => err(res, 500, String(e)));
  });
  server.on('error', e => {
    vscode.window.showErrorMessage(`AI Proxy Remote Chat: ${e.message}`);
    if (statusBar) {
      statusBar.text = '$(error) ProxyChat';
      statusBar.tooltip = `Listener error: ${e.message}`;
    }
  });
  server.listen(c.port, c.bind, () => {
    if (statusBar) {
      statusBar.text = `$(broadcast) ProxyChat ${c.bind}:${c.port}`;
      statusBar.tooltip = `Listening on http://${c.bind}:${c.port}/chat — POST {"prompt":"..."} to drive chat`;
    }
    console.log(`AI Proxy Remote Chat listening on ${c.bind}:${c.port}`);
    // Self-register with the AI Proxy so the phone PWA can route prompts here. Heartbeat
    // every 5 min keeps last_seen fresh so the proxy can detect stale entries.
    if (registerTimer) { clearInterval(registerTimer); registerTimer = null; }
    selfRegister().catch(() => {});
    registerTimer = setInterval(() => selfRegister().catch(() => {}), 5 * 60 * 1000);
  });
}

// ──────── Chat participant: @proxy with phone-approved tools ────────

const sleepP = (ms) => new Promise(r => setTimeout(r, ms));

async function _proxyApprove(toolName, args, summary) {
  const c = cfg();
  if (!c.proxyUrl) {
    return { decided: 'deny', reason: 'aiProxyRemoteChat.proxyUrl not set' };
  }
  const proxyBase = c.proxyUrl.replace(/\/+$/, '');
  // Register the pending tool call with the proxy.
  let pendingId;
  try {
    const r = await _post(proxyBase + '/__proxy/api/control/pending-tool', {
      tool_name: toolName,
      arguments: args,
      summary,
      source: c.advertiseName || os.hostname(),
    });
    if (r.status >= 200 && r.status < 300) {
      const j = JSON.parse(r.text);
      // Auto-decided by a persistent rule? Skip the wait.
      if (j.decision === 'allow' || j.decision === 'deny') {
        return { decided: j.decision, reason: 'auto: ' + (j.auto_rule || 'rule') };
      }
      pendingId = j.id;
    } else {
      return { decided: 'deny', reason: `proxy register failed: HTTP ${r.status}` };
    }
  } catch (e) {
    return { decided: 'deny', reason: `proxy unreachable: ${e.message}` };
  }
  // Poll for the user's decision.
  const pollMs = Math.max(200, Math.round((vscode.workspace.getConfiguration('aiProxyRemoteChat').get('participantPollInterval', 1.0)) * 1000));
  const timeoutMs = Math.max(5000, vscode.workspace.getConfiguration('aiProxyRemoteChat').get('participantApprovalTimeout', 120) * 1000);
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    await sleepP(pollMs);
    try {
      const r = await new Promise((resolve, reject) => {
        const u = url.parse(proxyBase + '/__proxy/api/control/pending-tool/' + pendingId);
        const lib = u.protocol === 'https:' ? https : http;
        const req = lib.request({ method: 'GET', protocol: u.protocol, hostname: u.hostname, port: u.port, path: u.path, timeout: 5000 }, (res) => {
          const chunks = [];
          res.on('data', c => chunks.push(c));
          res.on('end', () => resolve({ status: res.statusCode, text: Buffer.concat(chunks).toString('utf8') }));
        });
        req.on('error', reject);
        req.on('timeout', () => req.destroy(new Error('timeout')));
        req.end();
      });
      if (r.status === 200) {
        const j = JSON.parse(r.text);
        if (j.decision === 'allow' || j.decision === 'deny') {
          return { decided: j.decision, reason: 'phone' };
        }
      }
    } catch { /* keep polling */ }
  }
  return { decided: 'deny', reason: 'timed out waiting for phone approval' };
}

// ──── Tool implementations ────
async function _toolBash(args) {
  const command = (args && args.command) || '';
  if (!command || typeof command !== 'string') {
    return { ok: false, error: "missing 'command'" };
  }
  return new Promise((resolve) => {
    const cwd = (args.cwd && typeof args.cwd === 'string') ? args.cwd
      : (vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders[0]?.uri.fsPath);
    const child = require('child_process');
    const proc = child.spawn(command, { shell: true, cwd, windowsHide: true });
    let stdout = '', stderr = '';
    proc.stdout.on('data', d => { stdout += d.toString('utf8'); if (stdout.length > 50000) stdout = stdout.slice(-50000); });
    proc.stderr.on('data', d => { stderr += d.toString('utf8'); if (stderr.length > 50000) stderr = stderr.slice(-50000); });
    proc.on('close', (code) => resolve({ ok: code === 0, exit_code: code, stdout, stderr }));
    proc.on('error', (e) => resolve({ ok: false, error: String(e) }));
    setTimeout(() => { try { proc.kill(); } catch {} }, 2 * 60 * 1000);
  });
}

async function _toolReadFile(args) {
  const p = (args && (args.path || args.filePath)) || '';
  if (!p) return { ok: false, error: "missing 'path'" };
  try {
    const u = vscode.Uri.file(p);
    const buf = await vscode.workspace.fs.readFile(u);
    const text = Buffer.from(buf).toString('utf8');
    return { ok: true, content: text.length > 100000 ? text.slice(0, 100000) + '\n…[truncated]' : text };
  } catch (e) {
    return { ok: false, error: String(e) };
  }
}

const PARTICIPANT_TOOLS = [
  {
    name: 'bash',
    description: 'Run a shell command on the user\'s machine. Output capped at 50KB stdout/stderr.',
    inputSchema: { type: 'object', properties: { command: { type: 'string' }, cwd: { type: 'string' } }, required: ['command'] },
    impl: _toolBash,
    summary: (a) => `bash: ${(a.command || '').slice(0, 100)}`,
  },
  {
    name: 'read_file',
    description: 'Read a text file from the user\'s machine. Truncated to 100KB.',
    inputSchema: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] },
    impl: _toolReadFile,
    summary: (a) => `read_file: ${a.path || a.filePath || '?'}`,
  },
];
const PARTICIPANT_TOOL_BY_NAME = Object.fromEntries(PARTICIPANT_TOOLS.map(t => [t.name, t]));

function _toLmTools() {
  return PARTICIPANT_TOOLS.map(t => ({
    name: t.name,
    description: t.description,
    inputSchema: t.inputSchema,
  }));
}

async function _runParticipantTurn(request, context, stream, token) {
  // Build chat history.
  const history = [];
  for (const turn of (context.history || [])) {
    if (turn instanceof vscode.ChatRequestTurn) {
      history.push(vscode.LanguageModelChatMessage.User(turn.prompt));
    } else if (turn instanceof vscode.ChatResponseTurn) {
      let text = '';
      for (const r of (turn.response || [])) {
        if (r instanceof vscode.ChatResponseMarkdownPart) text += r.value.value;
      }
      if (text) history.push(vscode.LanguageModelChatMessage.Assistant(text));
    }
  }
  history.push(vscode.LanguageModelChatMessage.User(request.prompt));

  // Pick a model — Copilot first, fall back to anything available.
  let models = [];
  try {
    models = await vscode.lm.selectChatModels({ vendor: 'copilot' });
    if (!models.length) models = await vscode.lm.selectChatModels();
  } catch (e) {
    stream.markdown(`\n**Error:** could not access a language model: ${e}\n`);
    return;
  }
  if (!models.length) {
    stream.markdown('\n**Error:** no chat models available. Sign in to Copilot or install a chat model provider.\n');
    return;
  }
  const model = models[0];
  stream.progress(`Using ${model.id}…`);

  // Multi-iteration loop: model emits tool calls, we approve+execute, feed results back.
  let messages = history.slice();
  for (let iter = 0; iter < 8; iter++) {
    if (token.isCancellationRequested) return;
    let lmResp;
    try {
      lmResp = await model.sendRequest(messages, { tools: _toLmTools() }, token);
    } catch (e) {
      stream.markdown(`\n**Error from model:** ${e}\n`);
      return;
    }

    const toolCalls = [];
    let assistantText = '';
    for await (const part of lmResp.stream) {
      if (token.isCancellationRequested) return;
      if (part instanceof vscode.LanguageModelTextPart) {
        stream.markdown(part.value);
        assistantText += part.value;
      } else if (part instanceof vscode.LanguageModelToolCallPart) {
        toolCalls.push(part);
      }
    }
    if (!toolCalls.length) return;  // Done — model emitted only text.

    // Continue with the assistant's tool-call message in history.
    messages.push(vscode.LanguageModelChatMessage.Assistant(
      assistantText ? [new vscode.LanguageModelTextPart(assistantText), ...toolCalls] : toolCalls
    ));

    // For each tool call, approve via phone, execute, append result.
    const toolResultParts = [];
    for (const call of toolCalls) {
      const tool = PARTICIPANT_TOOL_BY_NAME[call.name];
      if (!tool) {
        stream.markdown(`\n_Skipping unknown tool: \`${call.name}\`_\n`);
        toolResultParts.push(new vscode.LanguageModelToolResultPart(call.callId, [
          new vscode.LanguageModelTextPart(`Error: unknown tool '${call.name}'`),
        ]));
        continue;
      }
      const summary = (tool.summary && tool.summary(call.input)) || `${call.name}(...)`;
      stream.markdown(`\n🔐 _Awaiting phone approval:_ \`${summary}\`\n`);
      const decision = await _proxyApprove(call.name, call.input || {}, summary);
      if (decision.decided !== 'allow') {
        stream.markdown(`\n❌ _Tool call denied (${decision.reason}). Telling the model._\n`);
        toolResultParts.push(new vscode.LanguageModelToolResultPart(call.callId, [
          new vscode.LanguageModelTextPart(`User denied this tool call (${decision.reason}).`),
        ]));
        continue;
      }
      stream.markdown(`\n✅ _Approved (${decision.reason}). Running…_\n`);
      try {
        const result = await tool.impl(call.input || {});
        const text = (typeof result === 'string') ? result : JSON.stringify(result, null, 2);
        toolResultParts.push(new vscode.LanguageModelToolResultPart(call.callId, [
          new vscode.LanguageModelTextPart(text),
        ]));
      } catch (e) {
        toolResultParts.push(new vscode.LanguageModelToolResultPart(call.callId, [
          new vscode.LanguageModelTextPart(`Tool error: ${e}`),
        ]));
      }
    }
    messages.push(vscode.LanguageModelChatMessage.User(toolResultParts));
  }
  stream.markdown('\n_(stopped after 8 tool-call iterations to avoid runaway loops.)_\n');
}

async function _participantHandler(request, context, stream, token) {
  try {
    await _runParticipantTurn(request, context, stream, token);
  } catch (e) {
    stream.markdown(`\n**Unhandled error:** ${e}\n`);
  }
}


function activate(context) {
  statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBar.command = 'aiProxyRemoteChat.showStatus';
  statusBar.show();
  context.subscriptions.push(statusBar);

  // Register the @proxy chat participant. Requires VS Code ≥1.95 for vscode.chat.createChatParticipant.
  try {
    if (vscode.chat && vscode.chat.createChatParticipant) {
      const participant = vscode.chat.createChatParticipant('aiproxy.chat', _participantHandler);
      participant.iconPath = new vscode.ThemeIcon('broadcast');
      context.subscriptions.push(participant);
      console.log('AI Proxy Remote Chat: registered @proxy participant');
    } else {
      console.warn('AI Proxy Remote Chat: vscode.chat API not available; skipping participant registration');
    }
  } catch (e) {
    console.warn('AI Proxy Remote Chat: failed to register chat participant: ' + e.message);
  }

  context.subscriptions.push(
    vscode.commands.registerCommand('aiProxyRemoteChat.showStatus', () => {
      const c = cfg();
      vscode.window.showInformationMessage(
        `Listening on http://${c.bind}:${c.port} · default command: ${c.defaultCommand}` +
        (c.token ? ' · token set' : ' · NO TOKEN')
      );
    }),
    vscode.commands.registerCommand('aiProxyRemoteChat.restart', () => {
      startServer();
      vscode.window.showInformationMessage('AI Proxy Remote Chat restarted.');
    }),
    vscode.workspace.onDidChangeConfiguration(e => {
      if (e.affectsConfiguration('aiProxyRemoteChat')) startServer();
    }),
  );
  startServer();
}

function deactivate() {
  if (registerTimer) { clearInterval(registerTimer); registerTimer = null; }
  if (server) { try { server.close(); } catch {} server = null; }
}

module.exports = { activate, deactivate };
