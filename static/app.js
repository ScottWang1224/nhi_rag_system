const chatForm = document.getElementById("chat-form");
const submitButton = document.getElementById("submit");
const queryInput = document.getElementById("query");
const status = document.getElementById("status");
const chatRoom = document.getElementById("chat-room");

const STREAM_DELAY_MS = 120;
const ASSISTANT_NAME = "\u5065\u4fdd\u554f\u7b54\u52a9\u624b";
const USER_NAME = "\u4f60";
const REFERENCE_TITLE = "\u53c3\u8003\u8cc7\u6599";
const EMPTY_QUERY_MESSAGE = "\u8acb\u5148\u8f38\u5165\u554f\u984c\u3002";
const LOADING_MESSAGE = "\u7cfb\u7d71\u6b63\u5728\u6574\u7406\u56de\u7b54...";
const REQUEST_ERROR_MESSAGE =
  "\u7cfb\u7d71\u66ab\u6642\u7121\u6cd5\u56de\u61c9\uff0c\u8acb\u7a0d\u5f8c\u518d\u8a66\u3002";
const ERROR_PREFIX = "\u767c\u751f\u932f\u8aa4\uff1a";
const FAILED_STATUS_MESSAGE = "\u9019\u6b21\u67e5\u8a62\u6c92\u6709\u6210\u529f\u5b8c\u6210\u3002";
const FEEDBACK_PROMPT = "\u9019\u5247\u56de\u8986\u5c0d\u4f60\u6709\u5e6b\u52a9\u55ce\uff1f";
const FEEDBACK_HELPFUL = "\u6709\u5e6b\u52a9";
const FEEDBACK_NOT_HELPFUL = "\u6c92\u6709\u5e6b\u52a9";
const FEEDBACK_THANKS = "\u5df2\u6536\u5230\u4f60\u7684\u56de\u994b\u3002";
const FEEDBACK_ERROR = "\u56de\u994b\u9001\u51fa\u5931\u6557\uff0c\u8acb\u7a0d\u5f8c\u518d\u8a66\u3002";

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function scrollToBottom() {
  chatRoom.scrollTop = chatRoom.scrollHeight;
}

function normalizeAnswerText(text) {
  const normalized = String(text).replace(/\r\n/g, "\n").trim();
  const markers = [
    "\n資料來源",
    "\n参考資料",
    "\n參考資料",
    "\nSources",
    "\nSource",
  ];

  let cutIndex = normalized.length;
  for (const marker of markers) {
    const index = normalized.indexOf(marker);
    if (index !== -1 && index < cutIndex) {
      cutIndex = index;
    }
  }

  return normalized.slice(0, cutIndex).trim();
}

function appendMessage({ role, sender, text, references = [] }) {
  const wrapper = document.createElement("article");
  wrapper.className = `message message-${role}`;

  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.textContent = sender;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  wrapper.appendChild(meta);
  wrapper.appendChild(bubble);

  if (role === "assistant" && references.length > 0) {
    const referenceBox = document.createElement("div");
    referenceBox.className = "references";

    const title = document.createElement("p");
    title.className = "references-title";
    title.textContent = REFERENCE_TITLE;
    referenceBox.appendChild(title);

    const list = document.createElement("ol");
    list.className = "reference-list";

    for (const reference of references) {
      const item = document.createElement("li");
      item.innerHTML =
        `<a href="${escapeHtml(reference.url)}" target="_blank" rel="noreferrer">` +
        `${escapeHtml(reference.title)}</a>`;
      list.appendChild(item);
    }

    referenceBox.appendChild(list);
    bubble.appendChild(referenceBox);
  }

  chatRoom.appendChild(wrapper);
  scrollToBottom();
}

function createAssistantMessage(sender) {
  const wrapper = document.createElement("article");
  wrapper.className = "message message-assistant";

  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.textContent = sender;

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  wrapper.appendChild(meta);
  wrapper.appendChild(bubble);
  chatRoom.appendChild(wrapper);
  scrollToBottom();

  return bubble;
}

function splitIntoChunks(text) {
  const normalized = String(text).replace(/\r\n/g, "\n");
  const rawParts =
    normalized.match(/[^。\uFF01\uFF1F\n]+[。\uFF01\uFF1F]?|[^\S\n]*\n/g) || [normalized];
  const chunks = [];

  for (const part of rawParts) {
    if (!part) {
      continue;
    }

    if (part.includes("\n")) {
      chunks.push(part);
      continue;
    }

    if (part.length <= 24) {
      chunks.push(part);
      continue;
    }

    const segments = part.match(/.{1,24}/g) || [part];
    chunks.push(...segments);
  }

  return chunks;
}

function sleep(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

async function submitFeedback({ requestId, feedback, feedbackBox }) {
  const buttons = feedbackBox.querySelectorAll("button");
  const statusNode = feedbackBox.querySelector(".feedback-status");

  for (const button of buttons) {
    button.disabled = true;
  }
  statusNode.textContent = "";

  try {
    const response = await fetch(`/api/query-logs/${encodeURIComponent(requestId)}/feedback`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_feedback: feedback }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || FEEDBACK_ERROR);
    }

    for (const button of buttons) {
      button.classList.toggle("selected", button.dataset.feedback === feedback);
    }
    statusNode.textContent = FEEDBACK_THANKS;
  } catch (error) {
    statusNode.textContent = FEEDBACK_ERROR;
  } finally {
    for (const button of buttons) {
      button.disabled = false;
    }
  }
}

function appendFeedbackControls({ bubble, requestId }) {
  if (!requestId) {
    console.warn("Feedback controls were not rendered because request_id is missing.");
    return;
  }

  const feedbackBox = document.createElement("div");
  feedbackBox.className = "feedback";

  const prompt = document.createElement("p");
  prompt.className = "feedback-prompt";
  prompt.textContent = FEEDBACK_PROMPT;
  feedbackBox.appendChild(prompt);

  const actions = document.createElement("div");
  actions.className = "feedback-actions";

  const helpfulButton = document.createElement("button");
  helpfulButton.type = "button";
  helpfulButton.dataset.feedback = "helpful";
  helpfulButton.textContent = FEEDBACK_HELPFUL;

  const notHelpfulButton = document.createElement("button");
  notHelpfulButton.type = "button";
  notHelpfulButton.dataset.feedback = "not_helpful";
  notHelpfulButton.textContent = FEEDBACK_NOT_HELPFUL;

  actions.appendChild(helpfulButton);
  actions.appendChild(notHelpfulButton);
  feedbackBox.appendChild(actions);

  const statusNode = document.createElement("p");
  statusNode.className = "feedback-status";
  feedbackBox.appendChild(statusNode);

  for (const button of [helpfulButton, notHelpfulButton]) {
    button.addEventListener("click", () => {
      submitFeedback({
        requestId,
        feedback: button.dataset.feedback,
        feedbackBox,
      });
    });
  }

  bubble.appendChild(feedbackBox);
  scrollToBottom();
}

async function typeAssistantMessage({ sender, text, references = [], requestId = "" }) {
  const bubble = createAssistantMessage(sender);
  const chunks = splitIntoChunks(normalizeAnswerText(text));

  for (const chunk of chunks) {
    const chunkNode = document.createElement("span");
    chunkNode.className = "stream-chunk";
    chunkNode.textContent = chunk;
    bubble.appendChild(chunkNode);
    scrollToBottom();
    await sleep(STREAM_DELAY_MS);
  }

  if (references.length > 0) {
    const referenceBox = document.createElement("div");
    referenceBox.className = "references";

    const title = document.createElement("p");
    title.className = "references-title";
    title.textContent = REFERENCE_TITLE;
    referenceBox.appendChild(title);

    const list = document.createElement("ol");
    list.className = "reference-list";

    for (const reference of references) {
      const item = document.createElement("li");
      item.innerHTML =
        `<a href="${escapeHtml(reference.url)}" target="_blank" rel="noreferrer">` +
        `${escapeHtml(reference.title)}</a>`;
      list.appendChild(item);
    }

    referenceBox.appendChild(list);
    bubble.appendChild(referenceBox);
    scrollToBottom();
  }

  appendFeedbackControls({ bubble, requestId });
}

async function sendQuery() {
  const query = queryInput.value.trim();
  if (!query) {
    status.textContent = EMPTY_QUERY_MESSAGE;
    return;
  }

  appendMessage({ role: "user", sender: USER_NAME, text: query });
  queryInput.value = "";
  queryInput.style.height = "";
  status.textContent = LOADING_MESSAGE;
  submitButton.disabled = true;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || REQUEST_ERROR_MESSAGE);
    }

    await typeAssistantMessage({
      sender: ASSISTANT_NAME,
      text: data.answer,
      references: data.references || [],
      requestId: data.request_id || data.requestId || "",
    });
    status.textContent = "";
  } catch (error) {
    appendMessage({
      role: "assistant",
      sender: ASSISTANT_NAME,
      text: `${ERROR_PREFIX}${error.message}`,
    });
    status.textContent = FAILED_STATUS_MESSAGE;
  } finally {
    submitButton.disabled = false;
    queryInput.focus();
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await sendQuery();
});

queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

queryInput.addEventListener("input", () => {
  queryInput.style.height = "auto";
  queryInput.style.height = `${Math.min(queryInput.scrollHeight, 180)}px`;
});
