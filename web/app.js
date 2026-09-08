const promptEl = document.querySelector('#prompt');
const resultEl = document.querySelector('#result');
const statusEl = document.querySelector('#model-status');
const statsEl = document.querySelector('#stats');
const runEl = document.querySelector('#run');
const copyEl = document.querySelector('#copy');
const creativeEl = document.querySelector('#creative');

let model = null;
let irModel = null;
let creativeMode = false;
const KEYWORDS = new Set('and as assert async await break case class continue def del elif else except finally for from global if import in is lambda match nonlocal not or pass raise return try while with yield True False None'.split(' '));
const BUILTINS = new Set('print len range int str float list dict set sum min max'.split(' '));

function normalTokenize(text) {
  return (text.toLocaleLowerCase().match(/[\p{L}\p{N}_]+/gu) || []);
}

function conceptTokens(prompt) {
  const tokens = [];
  for (const word of normalTokenize(prompt)) {
    if (['setze', 'speichere', 'set', 'save'].includes(word)) tokens.push('CONCEPT:SET');
    else if (['addiert', 'addiere', 'add', 'plus'].includes(word)) tokens.push('CONCEPT:ADD');
    else if (['subtrahiert', 'subtrahiere', 'sub', 'minus'].includes(word)) tokens.push('CONCEPT:SUB');
    else if (['multipliziert', 'multipliziere', 'multiply', 'mul', 'times'].includes(word)) tokens.push('CONCEPT:MUL');
    else if (['teilt', 'teile', 'divide', 'div', 'geteilt'].includes(word)) tokens.push('CONCEPT:DIV');
    else if (['schleife', 'loop', 'for'].includes(word)) tokens.push('CONCEPT:FOR_RANGE');
    else if (['aus', 'gib', 'gibt', 'print', 'output', 'ausgabe'].includes(word)) tokens.push('CONCEPT:PRINT');
    else if (/^\d+$/.test(word)) tokens.push(`NUM:${word}`);
    else if (['x', 'total', 'wert', 'i', 'number', 'item'].includes(word)) tokens.push(`VAR:${word}`);
  }
  return tokens;
}

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
    if (token.startsWith('KW:') || token.startsWith('FN:')) current.push(token.slice(3));
    else if (token.startsWith('OP:')) { const op = token.slice(3); if ([',', ':', ')', ']', '}'].includes(op) && current.length) current[current.length - 1] += op; else current.push(op); }
    else if (token === 'NAME:<id>') current.push('value');
    else if (token === 'LIT:<str>') current.push('"text"');
    else if (token === 'LIT:<num>') current.push('0');
  }
  if (current.length) lines.push('    '.repeat(indent) + current.join(' ').trim());
  return lines.join('\n');
}

function codeContinuation(source, maxTokens = 42) {
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

function recognizeIntent(prompt) {
  const stopwords = new Set(['a', 'an', 'the', 'in', 'on', 'of', 'to', 'and', 'ein', 'eine', 'einen', 'der', 'die', 'das', 'für', 'mit', 'und', 'zu']);
  const words = new Set(normalTokenize(prompt).filter(word => !stopwords.has(word)));
  let bestIntent = null;
  let bestScore = 0;
  for (const [intent, pattern] of Object.entries(model.intent_patterns || {})) {
    const matches = pattern.filter(word => words.has(word));
    const score = matches.reduce((sum, word) => sum + 1 / (1 + pattern.indexOf(word)), 0);
    if (score > bestScore) { bestScore = score; bestIntent = intent; }
  }
  return bestIntent;
}

function generateForPrompt(prompt) {
  const intent = recognizeIntent(prompt);
  if (intent && model.templates?.[intent]) {
    const candidates = Array.isArray(model.templates[intent]) ? model.templates[intent] : [model.templates[intent]];
    const words = new Set(normalTokenize(prompt));
    const german = ['hallo', 'welt', 'schreibe', 'beispiel', 'begrüße', 'begrüßen', 'summe', 'schleife'].some(word => words.has(word));
    const preferred = candidates.filter(candidate => candidate.includes('Hallo') === german);
    return { intent, code: (preferred.length ? preferred : candidates)[0] };
  }
  return { intent: null, code: codeContinuation(prompt) };
}

function irNext(history) {
  for (let width = Math.min(irModel.order - 1, history.length); width > 0; width--) {
    const context = history.slice(-width).join('␟');
    const entries = Object.entries(irModel.transitions[context] || {}).sort((a, b) => b[1] - a[1]);
    if (entries.length) return entries[0][0];
  }
  return '<END>';
}

function compileIR(tokens) {
  const lines = [];
  let indent = 0;
  const value = raw => raw.split(':').slice(1).join(':').replaceAll('_', ' ');
  for (let i = 0; i < tokens.length;) {
    const token = tokens[i];
    if (['<END>', '<NL>', '<BOS>'].includes(token)) { i++; continue; }
    const prefix = '    '.repeat(indent);
    if (token === 'SET') { lines.push(`${prefix}${value(tokens[i + 1])} = ${value(tokens[i + 2])}`); i += 3; }
    else if (['ADD', 'SUB', 'MUL', 'DIV'].includes(token)) { const op = {ADD: '+', SUB: '-', MUL: '*', DIV: '/'}[token]; lines.push(`${prefix}${value(tokens[i + 1])} = ${value(tokens[i + 1])} ${op} ${value(tokens[i + 2])}`); i += 3; }
    else if (token === 'PRINT') { lines.push(`${prefix}print(${value(tokens[i + 1])})`); i += 2; }
    else if (token === 'FOR_RANGE') { lines.push(`${prefix}for ${value(tokens[i + 1])} in range(${value(tokens[i + 2])}):`); indent++; i += 3; }
    else if (token === 'END_FOR') { indent = Math.max(0, indent - 1); i++; }
    else i++;
  }
  return lines.join('\n') + (lines.length ? '\n' : '');
}

function generateCreative(prompt) {
  const history = ['<BOS>', '<PROMPT>', ...conceptTokens(prompt), '<IR>'];
  const output = [];
  for (let index = 0; index < 48; index++) {
    const token = irNext(history);
    output.push(token); history.push(token);
    if (token === '<END>') break;
  }
  const code = compileIR(output);
  return { intent: 'creative-ir', code: code || '# Keine gültige IR-Fortsetzung gefunden. Bitte Prompt vereinfachen.' };
}

async function loadModel() {
  try {
    const response = await fetch('model.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`model.json returned ${response.status}`);
    model = await response.json();
    const irResponse = await fetch('ir_model.json', { cache: 'no-store' });
    if (irResponse.ok) irModel = await irResponse.json();
    const intentCount = Object.keys(model.intent_patterns || {}).length;
    statusEl.textContent = `model ready · ${model.vocab.length} code tokens · ${intentCount} prompt intents`;
    statsEl.textContent = `${Object.keys(model.transitions).length} code contexts · ${model.prompt_vocab?.length || 0} text words · ${irModel ? 'IR ready' : 'IR unavailable'}`;
  } catch (error) {
    statusEl.textContent = 'model not found · run train.py';
    statusEl.classList.add('error');
    resultEl.textContent = 'The model artifact is missing. Run `python3 train.py` and reload.';
  }
}

function run() {
  if (!model) return;
  const prompt = promptEl.value.trim();
  const result = creativeMode && irModel ? generateCreative(prompt) : generateForPrompt(prompt || 'def greet(name):\n    ');
  resultEl.textContent = result.code;
  statsEl.textContent = result.intent ? `intent matched · ${result.intent}` : 'code-prefix fallback · no intent matched';
}

runEl.addEventListener('click', run);
creativeEl.addEventListener('click', () => {
  creativeMode = !creativeMode;
  creativeEl.textContent = `Creative IR: ${creativeMode ? 'on' : 'off'}`;
});
promptEl.addEventListener('keydown', event => { if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') run(); });
copyEl.addEventListener('click', async () => { await navigator.clipboard.writeText(resultEl.textContent); copyEl.textContent = 'Copied'; setTimeout(() => copyEl.textContent = 'Copy', 1200); });
loadModel();
