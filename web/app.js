const promptEl = document.querySelector('#prompt');
const resultEl = document.querySelector('#result');
const statusEl = document.querySelector('#model-status');
const statsEl = document.querySelector('#stats');
const runEl = document.querySelector('#run');
const copyEl = document.querySelector('#copy');

let model = null;
const KEYWORDS = new Set('and as assert async await break case class continue def del elif else except finally for from global if import in is lambda match nonlocal not or pass raise return try while with yield True False None'.split(' '));
const BUILTINS = new Set('print len range int str float list dict set sum min max'.split(' '));

function tokenizePython(source) {
  const tokens = ['<BOS>'];
  const lines = source.replaceAll('\r', '').split('\n');
  let previousIndent = 0;
  for (const line of lines) {
    if (!line.trim()) continue;
    const leading = line.match(/^ */)[0].length;
    const indent = Math.floor(leading / 4);
    while (previousIndent < indent) { tokens.push('<INDENT>'); previousIndent++; }
    while (previousIndent > indent) { tokens.push('<DEDENT>'); previousIndent--; }
    const parts = line.trim().match(/(?:"[^"\n]*"|'[^'\n]*'|\d+(?:\.\d+)?|[A-Za-z_]\w*|==|!=|<=|>=|\+=|-=|\*\*|\/\/|[^\s])/g) || [];
    for (const part of parts) {
      if (KEYWORDS.has(part)) tokens.push(`KW:${part}`);
      else if (BUILTINS.has(part)) tokens.push(`FN:${part}`);
      else if (/^[A-Za-z_]\w*$/.test(part)) tokens.push('NAME:<id>');
      else if (/^['"]/.test(part)) tokens.push('LIT:<str>');
      else if (/^\d/.test(part)) tokens.push('LIT:<num>');
      else tokens.push(`OP:${part}`);
    }
    tokens.push('<NEWLINE>');
    previousIndent = indent;
  }
  while (previousIndent > 0) { tokens.push('<DEDENT>'); previousIndent--; }
  return tokens;
}

function nextToken(history) {
  const order = model.order;
  const context = history.slice(-(order - 1)).join('␟') || '<BOS>';
  const candidates = model.transitions[context] || model.transitions[history.at(-1)] || {};
  const entries = Object.entries(candidates);
  if (!entries.length) return '<END>';
  entries.sort((a, b) => b[1] - a[1]);
  return entries[0][0];
}

function detokenize(tokens) {
  const lines = [];
  let current = [];
  let indent = 0;
  for (const token of tokens) {
    if (token === '<BOS>' || token === '<END>') continue;
    if (token === '<INDENT>') { indent++; continue; }
    if (token === '<DEDENT>') { if (current.length) lines.push('    '.repeat(indent) + current.join(' ').trim()); current = []; indent = Math.max(0, indent - 1); continue; }
    if (token === '<NEWLINE>') { if (current.length) lines.push('    '.repeat(indent) + current.join(' ').trim()); current = []; continue; }
    if (token.startsWith('KW:')) current.push(token.slice(3));
    else if (token.startsWith('FN:')) current.push(token.slice(3));
    else if (token.startsWith('OP:')) { const op = token.slice(3); if ([',', ':', ')', ']', '}'].includes(op) && current.length) current[current.length - 1] += op; else current.push(op); }
    else if (token === 'NAME:<id>') current.push('value');
    else if (token === 'LIT:<str>') current.push('"text"');
    else if (token === 'LIT:<num>') current.push('0');
  }
  if (current.length) lines.push('    '.repeat(indent) + current.join(' ').trim());
  return lines.join('\n');
}

function generate(source, maxTokens = 42) {
  const history = tokenizePython(source);
  const generated = [];
  for (let index = 0; index < maxTokens; index++) {
    const token = nextToken(history);
    generated.push(token);
    history.push(token);
    if (token === '<END>') break;
  }
  return `${source.trimEnd()}\n${detokenize(generated)}`.trim();
}

async function loadModel() {
  try {
    const response = await fetch('model.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`model.json returned ${response.status}`);
    model = await response.json();
    statusEl.textContent = `model ready · ${model.vocab.length} tokens`;
    statsEl.textContent = `${Object.keys(model.transitions).length} contexts · order ${model.order}`;
  } catch (error) {
    statusEl.textContent = 'model not found · run train.py';
    statusEl.classList.add('error');
    resultEl.textContent = 'The model artifact is missing. Run `python3 train.py` and reload.';
  }
}

function run() {
  if (!model) return;
  resultEl.textContent = generate(promptEl.value || 'def greet(name):\n    ');
}

runEl.addEventListener('click', run);
promptEl.addEventListener('keydown', event => { if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') run(); });
copyEl.addEventListener('click', async () => { await navigator.clipboard.writeText(resultEl.textContent); copyEl.textContent = 'Copied'; setTimeout(() => copyEl.textContent = 'Copy', 1200); });
loadModel();
