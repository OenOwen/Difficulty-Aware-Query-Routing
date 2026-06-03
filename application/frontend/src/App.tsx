import { useState } from "react";
import type { FormEvent, ReactNode } from "react";

type ChatResponse = {
  answer: string;
  time_taken: number;
  cost: number;
  model_size: string;
  model?: string;
  input_tokens?: number;
  output_tokens?: number;
  response_id?: string;
};

type FeedbackChoice = "up" | "down" | null;
type RoutedModel = "SLM" | "LLM";
type AppMode = "router" | "cascade" | null;
type UtilitySettingKey =
  | "LAMBDA_LATENCY"
  | "C_ERR"
  | "MANUAL_Q_S_SIMPLE"
  | "MANUAL_Q_S_COMPLEX"
  | "MANUAL_Q_L_SIMPLE"
  | "MANUAL_Q_L_COMPLEX"
  | "SLM_AVG_LATENCY"
  | "LLM_AVG_LATENCY"
  | "LLM_AVG_INPUT_TOKENS"
  | "LLM_AVG_OUTPUT_TOKENS";

type UtilitySettings = Record<UtilitySettingKey, number>;

const gpt5InputPricePer1M = 1.25;
const gpt5OutputPricePer1M = 10;
const muCost = 100;

const defaultUtilitySettings: UtilitySettings = {
  LAMBDA_LATENCY: 0.11,
  C_ERR: 15,
  MANUAL_Q_S_SIMPLE: 0.76,
  MANUAL_Q_S_COMPLEX: 0.39,
  MANUAL_Q_L_SIMPLE: 0.93,
  MANUAL_Q_L_COMPLEX: 0.83,
  SLM_AVG_LATENCY: 3.1,
  LLM_AVG_LATENCY: 23.81,
  LLM_AVG_INPUT_TOKENS: 154,
  LLM_AVG_OUTPUT_TOKENS: 1897.18,
};

const utilitySettingKeys = Object.keys(defaultUtilitySettings) as UtilitySettingKey[];

function computeExpectedCost(inputTokens: number, outputTokens: number) {
  return (
    (inputTokens / 1_000_000) * gpt5InputPricePer1M +
    (outputTokens / 1_000_000) * gpt5OutputPricePer1M
  );
}

function computeThreshold(settings: UtilitySettings) {
  const deltaC =
    computeExpectedCost(settings.LLM_AVG_INPUT_TOKENS, settings.LLM_AVG_OUTPUT_TOKENS) *
      muCost +
    settings.LAMBDA_LATENCY * (settings.LLM_AVG_LATENCY - settings.SLM_AVG_LATENCY);
  const dSimple = settings.MANUAL_Q_L_SIMPLE - settings.MANUAL_Q_S_SIMPLE;
  const dComplex = settings.MANUAL_Q_L_COMPLEX - settings.MANUAL_Q_S_COMPLEX;

  return (deltaC / settings.C_ERR - dSimple) / (dComplex - dSimple);
}

function readLatexGroup(text: string, startIndex: number) {
  if (text[startIndex] !== "{") {
    return { value: "", nextIndex: startIndex };
  }

  let depth = 0;
  let value = "";
  for (let index = startIndex; index < text.length; index += 1) {
    const char = text[index];
    if (char === "{") {
      if (depth > 0) value += char;
      depth += 1;
      continue;
    }

    if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        return { value, nextIndex: index + 1 };
      }
      value += char;
      continue;
    }

    value += char;
  }

  return { value, nextIndex: text.length };
}

function renderLatexExpression(expression: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  let index = 0;

  while (index < expression.length) {
    if (expression.startsWith("\\frac", index)) {
      const numerator = readLatexGroup(expression, index + "\\frac".length);
      const denominator = readLatexGroup(expression, numerator.nextIndex);
      nodes.push(
        <span className="math-frac" key={nodes.length}>
          <span className="math-frac-top">{renderLatexExpression(numerator.value)}</span>
          <span className="math-frac-bottom">{renderLatexExpression(denominator.value)}</span>
        </span>,
      );
      index = denominator.nextIndex;
      continue;
    }

    if (expression.startsWith("\\text", index)) {
      const textGroup = readLatexGroup(expression, index + "\\text".length);
      nodes.push(<span key={nodes.length}>{textGroup.value}</span>);
      index = textGroup.nextIndex;
      continue;
    }

    if (expression[index] === "^" || expression[index] === "_") {
      const Tag = expression[index] === "^" ? "sup" : "sub";
      const nextIndex = index + 1;
      const group =
        expression[nextIndex] === "{"
          ? readLatexGroup(expression, nextIndex)
          : { value: expression[nextIndex] ?? "", nextIndex: nextIndex + 1 };
      nodes.push(<Tag key={nodes.length}>{renderLatexExpression(group.value)}</Tag>);
      index = group.nextIndex;
      continue;
    }

    if (expression[index] === "\\") {
      const commandMatch = expression.slice(index + 1).match(/^[A-Za-z]+/);
      if (commandMatch) {
        nodes.push(<span key={nodes.length}>{commandMatch[0]}</span>);
        index += commandMatch[0].length + 1;
        continue;
      }
    }

    nodes.push(expression[index]);
    index += 1;
  }

  return nodes;
}

function renderInlineMath(text: string): ReactNode[] {
  return text.split(/(\\\(.+?\\\))/g).map((part, index) => {
    if (part.startsWith("\\(") && part.endsWith("\\)")) {
      return (
        <span className="math-inline" key={index}>
          {renderLatexExpression(part.slice(2, -2))}
        </span>
      );
    }

    return part;
  });
}

function renderInlineMarkdown(text: string): ReactNode[] {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g);

  return parts.map((part, index) => {
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={index}>{part.slice(1, -1)}</code>;
    }

    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }

    if (part.startsWith("*") && part.endsWith("*")) {
      return <em key={index}>{part.slice(1, -1)}</em>;
    }

    return <span key={index}>{renderInlineMath(part)}</span>;
  });
}

function MarkdownText({ text }: { text: string }) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];

    if (!line.trim()) {
      index += 1;
      continue;
    }

    if (line.trim().startsWith("```")) {
      const codeLines: string[] = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }
      index += 1;
      blocks.push(
        <pre key={blocks.length}>
          <code>{codeLines.join("\n")}</code>
        </pre>,
      );
      continue;
    }

    const displayMathMatch = line.trim().match(/^\\\[(.*)\\\]$/);
    if (displayMathMatch) {
      blocks.push(
        <div className="math-display" key={blocks.length}>
          {renderLatexExpression(displayMathMatch[1])}
        </div>,
      );
      index += 1;
      continue;
    }

    const headingMatch = line.match(/^(#{1,3})\s+(.+)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const content = renderInlineMarkdown(headingMatch[2]);
      const HeadingTag = `h${level}` as "h1" | "h2" | "h3";
      blocks.push(<HeadingTag key={blocks.length}>{content}</HeadingTag>);
      index += 1;
      continue;
    }

    if (/^\s*[-*]\s+/.test(line)) {
      const items: ReactNode[] = [];
      while (index < lines.length && /^\s*[-*]\s+/.test(lines[index])) {
        items.push(<li key={items.length}>{renderInlineMarkdown(lines[index].replace(/^\s*[-*]\s+/, ""))}</li>);
        index += 1;
      }
      blocks.push(<ul key={blocks.length}>{items}</ul>);
      continue;
    }

    if (/^\s*\d+\.\s+/.test(line)) {
      const items: ReactNode[] = [];
      while (index < lines.length && /^\s*\d+\.\s+/.test(lines[index])) {
        items.push(<li key={items.length}>{renderInlineMarkdown(lines[index].replace(/^\s*\d+\.\s+/, ""))}</li>);
        index += 1;
      }
      blocks.push(<ol key={blocks.length}>{items}</ol>);
      continue;
    }

    const paragraphLines = [line];
    index += 1;
    while (
      index < lines.length &&
      lines[index].trim() &&
      !lines[index].trim().startsWith("```") &&
      !/^(#{1,3})\s+/.test(lines[index]) &&
      !/^\s*[-*]\s+/.test(lines[index]) &&
      !/^\s*\d+\.\s+/.test(lines[index])
    ) {
      paragraphLines.push(lines[index]);
      index += 1;
    }
    blocks.push(<p key={blocks.length}>{renderInlineMarkdown(paragraphLines.join(" "))}</p>);
  }

  return <div className="markdown-content">{blocks}</div>;
}

function App() {
  const [prompt, setPrompt] = useState("");
  const [mode, setMode] = useState<AppMode>(null);
  const [submittedPrompt, setSubmittedPrompt] = useState("");
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [cascadeSlmResponse, setCascadeSlmResponse] = useState<ChatResponse | null>(null);
  const [cascadeLlmResponse, setCascadeLlmResponse] = useState<ChatResponse | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [isEscalating, setIsEscalating] = useState(false);
  const [isSendingFeedback, setIsSendingFeedback] = useState(false);
  const [feedbackChoice, setFeedbackChoice] = useState<FeedbackChoice>(null);
  const [cascadeComplete, setCascadeComplete] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [useUtility, setUseUtility] = useState(true);
  const [utilitySettings, setUtilitySettings] = useState<UtilitySettings>(defaultUtilitySettings);
  const [sendingModel, setSendingModel] = useState<RoutedModel | null>(null);
  const utilityThreshold = computeThreshold(utilitySettings);
  const formattedThreshold = Number.isFinite(utilityThreshold)
    ? utilityThreshold.toFixed(4)
    : "Unavailable";
  const thresholdWarning =
    utilityThreshold < 0
      ? "Warning: All questions will now go to LLM"
      : utilityThreshold > 1
        ? "Warning: All questions will now go to SLM"
        : "";

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextPrompt = prompt.trim();
    if (!nextPrompt || isSending || response) return;

    setIsSending(true);
    setSendingModel(null);
    setErrorMessage("");

    try {
      const requestBody = {
        prompt: nextPrompt,
        use_utility: useUtility,
        utility_settings: utilitySettings,
      };
      const routeResponse = await fetch(`${import.meta.env.VITE_API_URL}/api/route`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      if (!routeResponse.ok) {
        throw new Error("Route request failed");
      }

      const routeData = (await routeResponse.json()) as { model?: RoutedModel };
      const selectedModel = routeData.model === "LLM" ? "LLM" : "SLM";
      setSendingModel(selectedModel);

      const apiResponse = await fetch(`${import.meta.env.VITE_API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...requestBody,
          selected_model: selectedModel,
        }),
      });

      if (!apiResponse.ok) {
        throw new Error("Request failed");
      }

      const data = (await apiResponse.json()) as Partial<ChatResponse>;
      setSubmittedPrompt(nextPrompt);
      setPrompt("");
      setResponse({
        answer: data.answer ?? "No answer returned from backend.",
        time_taken: data.time_taken ?? 0,
        cost: data.cost ?? 0,
        model_size: data.model_size ?? data.model ?? "small model",
        model: data.model,
        input_tokens: data.input_tokens,
        output_tokens: data.output_tokens,
        response_id: data.response_id,
      });
    } catch {
      setErrorMessage("Backend error. Check Flask logs and try again.");
    } finally {
      setIsSending(false);
      setSendingModel(null);
    }
  }

  async function handleCascadeSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextPrompt = prompt.trim();
    if (!nextPrompt || isSending || cascadeSlmResponse) return;

    setIsSending(true);
    setErrorMessage("");

    try {
      const apiResponse = await fetch(`${import.meta.env.VITE_API_URL}/api/cascade/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: nextPrompt }),
      });

      if (!apiResponse.ok) {
        throw new Error("Cascade request failed");
      }

      const data = (await apiResponse.json()) as Partial<ChatResponse>;
      setSubmittedPrompt(nextPrompt);
      setPrompt("");
      setCascadeSlmResponse({
        answer: data.answer ?? "No answer returned from backend.",
        time_taken: data.time_taken ?? 0,
        cost: data.cost ?? 0,
        model_size: data.model_size ?? data.model ?? "small model",
        model: data.model,
        input_tokens: data.input_tokens,
        output_tokens: data.output_tokens,
        response_id: data.response_id,
      });
    } catch {
      setErrorMessage("Backend error. Check Flask logs and try again.");
    } finally {
      setIsSending(false);
    }
  }

  async function handleFeedback(nextFeedback: Exclude<FeedbackChoice, null>) {
    if (!response || feedbackChoice || isSendingFeedback) return;
    setIsSendingFeedback(true);
    setErrorMessage("");

    try {
      const apiResponse = await fetch(`${import.meta.env.VITE_API_URL}/api/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          feedback: nextFeedback,
          question: submittedPrompt,
          answer: response.answer,
          latency: response.time_taken,
          input_tokens: response.input_tokens,
          output_tokens: response.output_tokens,
          model: response.model,
          response_id: response.response_id,
        }),
      });

      if (!apiResponse.ok) {
        throw new Error("Feedback failed");
      }

      setFeedbackChoice(nextFeedback);
    } catch {
      setErrorMessage("Could not send feedback. Try again.");
    } finally {
      setIsSendingFeedback(false);
    }
  }

  async function handleCascadeSatisfied() {
    if (!cascadeSlmResponse || cascadeComplete || isSendingFeedback) return;
    setIsSendingFeedback(true);
    setErrorMessage("");

    try {
      const apiResponse = await fetch(`${import.meta.env.VITE_API_URL}/api/cascade/satisfied`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: submittedPrompt,
          answer: cascadeSlmResponse.answer,
          latency: cascadeSlmResponse.time_taken,
          input_tokens: cascadeSlmResponse.input_tokens,
          output_tokens: cascadeSlmResponse.output_tokens,
          response_id: cascadeSlmResponse.response_id,
        }),
      });

      if (!apiResponse.ok) {
        throw new Error("Cascade feedback failed");
      }

      setCascadeComplete(true);
    } catch {
      setErrorMessage("Could not save cascading response. Try again.");
    } finally {
      setIsSendingFeedback(false);
    }
  }

  async function handleCascadeEscalate() {
    if (!cascadeSlmResponse || cascadeComplete || isEscalating) return;
    setIsEscalating(true);
    setErrorMessage("");

    try {
      const apiResponse = await fetch(`${import.meta.env.VITE_API_URL}/api/cascade/escalate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: submittedPrompt,
          slm_response_id: cascadeSlmResponse.response_id,
          slm_answer: cascadeSlmResponse.answer,
          slm_latency: cascadeSlmResponse.time_taken,
          slm_input_tokens: cascadeSlmResponse.input_tokens,
          slm_output_tokens: cascadeSlmResponse.output_tokens,
        }),
      });

      if (!apiResponse.ok) {
        throw new Error("Escalation failed");
      }

      const data = (await apiResponse.json()) as Partial<ChatResponse>;
      setCascadeLlmResponse({
        answer: data.answer ?? "No answer returned from backend.",
        time_taken: data.time_taken ?? 0,
        cost: data.cost ?? 0,
        model_size: data.model_size ?? data.model ?? "large model",
        model: data.model,
        input_tokens: data.input_tokens,
        output_tokens: data.output_tokens,
        response_id: data.response_id,
      });
      setCascadeComplete(true);
    } catch {
      setErrorMessage("Could not escalate to LLM. Try again.");
    } finally {
      setIsEscalating(false);
    }
  }

  function handleNextQuestion() {
    setPrompt("");
    setSubmittedPrompt("");
    setResponse(null);
    setCascadeSlmResponse(null);
    setCascadeLlmResponse(null);
    setFeedbackChoice(null);
    setCascadeComplete(false);
    setErrorMessage("");
  }

  function handleModeSelect(nextMode: Exclude<AppMode, null>) {
    handleNextQuestion();
    setMode(nextMode);
  }

  function handleHome() {
    handleNextQuestion();
    setMode(null);
  }

  function handleUtilitySettingChange(key: UtilitySettingKey, value: string) {
    setUtilitySettings((currentSettings) => ({
      ...currentSettings,
      [key]: value === "" ? 0 : Number(value),
    }));
  }

  return (
    <div className="app-shell">
      <header className="top-bar">
        <span>
          {mode === "cascade"
            ? "Cascading Router"
            : mode === "router"
              ? "Semantic Router"
              : "Router Selection"}
        </span>
        {mode && (
          <button type="button" className="home-btn" onClick={handleHome}>
            Home
          </button>
        )}
      </header>

      {!mode && (
        <main className="home-area">
          <button type="button" className="method-card" onClick={() => handleModeSelect("router")}>
            <strong>The Router</strong>
            <span>Use the current semantic router with utility settings and feedback logging.</span>
          </button>
          <button type="button" className="method-card" onClick={() => handleModeSelect("cascade")}>
            <strong>The Cascading Router</strong>
            <span>Send every question to the SLM first, then accept it or escalate to the LLM.</span>
          </button>
        </main>
      )}

      {mode === "router" && (
        <>
          <section className="settings-bar" aria-label="Router settings">
            <label className="utility-toggle">
              <input
                type="checkbox"
                checked={useUtility}
                onChange={(event) => setUseUtility(event.target.checked)}
                disabled={isSending}
              />
              <span>Use Utility</span>
            </label>

            <div className={`threshold-panel${thresholdWarning ? " warning" : ""}`}>
              <div className="threshold-value">
                <span>Threshold</span>
                <strong>{formattedThreshold}</strong>
              </div>
              {thresholdWarning && <p>{thresholdWarning}</p>}
            </div>

            {useUtility && (
              <div className="utility-settings-grid">
                {utilitySettingKeys.map((key) => (
                  <label className="setting-field" key={key}>
                    <span>{key}</span>
                    <input
                      type="number"
                      value={utilitySettings[key]}
                      onChange={(event) => handleUtilitySettingChange(key, event.target.value)}
                      step="any"
                      disabled={isSending}
                    />
                  </label>
                ))}
              </div>
            )}
          </section>
          <main className="chat-area">
            <form className="composer" onSubmit={handleSubmit}>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    e.currentTarget.form?.requestSubmit();
                  }
                }}
                placeholder="Ask your question..."
                rows={3}
                disabled={isSending || !!response}
              />
              <button type="submit" disabled={isSending || !!response || !prompt.trim()}>
                {isSending
                  ? sendingModel
                    ? `Sending to ${sendingModel}`
                    : "Selecting model..."
                  : "Send question"}
              </button>
            </form>

            {submittedPrompt && <div className="question-card">Question: {submittedPrompt}</div>}

            {response && (
              <section className="answer-section">
                <div className="answer-card">
                  <MarkdownText text={response.answer} />
                </div>
                <div className="meta-row">
                  <span>Time taken: {response.time_taken}</span>
                  <span>Cost: {response.cost}</span>
                  <span>Model: {response.model_size}</span>
                </div>

                {!feedbackChoice && (
                  <div className="feedback-row">
                    <button
                      type="button"
                      className="feedback-btn up"
                      onClick={() => handleFeedback("up")}
                      disabled={isSendingFeedback}
                    >
                      {"\u{1F44D}"}
                    </button>
                    <button
                      type="button"
                      className="feedback-btn down"
                      onClick={() => handleFeedback("down")}
                      disabled={isSendingFeedback}
                    >
                      {"\u{1F44E}"}
                    </button>
                  </div>
                )}

                {feedbackChoice && (
                  <button type="button" className="next-btn" onClick={handleNextQuestion}>
                    Next question
                  </button>
                )}
              </section>
            )}

            {errorMessage && <p className="error-text">{errorMessage}</p>}
          </main>
        </>
      )}

      {mode === "cascade" && (
        <main className="chat-area">
          <form className="composer" onSubmit={handleCascadeSubmit}>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  e.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder="Ask your question..."
              rows={3}
              disabled={isSending || isEscalating || !!cascadeSlmResponse}
            />
            <button
              type="submit"
              disabled={isSending || isEscalating || !!cascadeSlmResponse || !prompt.trim()}
            >
              {isSending ? "Sending to SLM" : "Send to SLM"}
            </button>
          </form>

          {submittedPrompt && <div className="question-card">Question: {submittedPrompt}</div>}

          {cascadeSlmResponse && (
            <section className="answer-section">
              <div className="answer-card">
                <MarkdownText text={cascadeSlmResponse.answer} />
              </div>
              <div className="meta-row">
                <span>Time taken: {cascadeSlmResponse.time_taken}</span>
                <span>Cost: {cascadeSlmResponse.cost}</span>
                <span>Model: {cascadeSlmResponse.model_size}</span>
              </div>

              {!cascadeComplete && (
                <div className="cascade-actions">
                  <button
                    type="button"
                    className="satisfied-btn"
                    onClick={handleCascadeSatisfied}
                    disabled={isSendingFeedback || isEscalating}
                  >
                    Satisfied
                  </button>
                  <button
                    type="button"
                    className="escalate-btn"
                    onClick={handleCascadeEscalate}
                    disabled={isSendingFeedback || isEscalating}
                  >
                    {isEscalating ? "Sending to LLM" : "Escalate to LLM"}
                  </button>
                </div>
              )}
            </section>
          )}

          {cascadeLlmResponse && (
            <section className="answer-section">
              <div className="answer-card">
                <MarkdownText text={cascadeLlmResponse.answer} />
              </div>
              <div className="meta-row">
                <span>Time taken: {cascadeLlmResponse.time_taken}</span>
                <span>Cost: {cascadeLlmResponse.cost}</span>
                <span>Model: {cascadeLlmResponse.model_size}</span>
              </div>
            </section>
          )}

          {cascadeComplete && (
            <button type="button" className="next-btn" onClick={handleNextQuestion}>
              Next question
            </button>
          )}

          {errorMessage && <p className="error-text">{errorMessage}</p>}
        </main>
      )}
    </div>
  );
}

export default App;
