const dropzone = document.getElementById("dropzone");
const dropzoneContent = document.getElementById("dropzoneContent");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const previewImg = document.getElementById("previewImg");
const analyzeBtn = document.getElementById("analyzeBtn");
const resetBtn = document.getElementById("resetBtn");
const loading = document.getElementById("loading");
const errorBox = document.getElementById("errorBox");
const result = document.getElementById("result");
const mascotAssistant = document.getElementById("mascotAssistant");
const mascotImg = document.getElementById("mascotImg");
const mascotBubble = document.getElementById("mascotBubble");

const MASCOT_STATES = {
  normal: { src: "/static/mascot-normal.png", cls: "", text: "สวัสดี! อัปโหลดรูปน้องหมาแล้วกดวิเคราะห์ได้เลย" },
  happy: { src: "/static/mascot-happy.png", cls: "is-happy", text: "น้องอารมณ์ดีมากเลย ดีใจด้วยนะ!" },
  angry: { src: "/static/mascot-angry.png", cls: "is-angry", text: "น้องดูไม่ค่อยพอใจ ลองดูคำแนะนำนะ" },
};

function setMascot(state) {
  const m = MASCOT_STATES[state] || MASCOT_STATES.normal;
  mascotImg.src = m.src;
  if (mascotAssistant) mascotAssistant.className = "mascot-assistant " + m.cls;
  // re-trigger the pop animation on every change
  mascotImg.style.animation = "none";
  void mascotImg.offsetWidth;
  mascotImg.style.animation = "";
  if (mascotBubble) mascotBubble.textContent = m.text;
}

let selectedFile = null;

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = false;
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

function selectFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("กรุณาเลือกไฟล์รูปภาพเท่านั้น");
    return;
  }
  clearError();
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.hidden = false;
  dropzoneContent.hidden = true;
  analyzeBtn.disabled = false;
  resetBtn.hidden = false;
  result.hidden = true;
}

browseBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});
dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => selectFile(e.target.files[0]));

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  selectFile(file);
});

resetBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  selectedFile = null;
  fileInput.value = "";
  previewImg.hidden = true;
  dropzoneContent.hidden = false;
  analyzeBtn.disabled = true;
  resetBtn.hidden = true;
  result.hidden = true;
  clearError();
  setMascot("normal");
});

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  clearError();
  result.hidden = true;
  loading.hidden = false;
  analyzeBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const res = await fetch("/api/predict", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "เกิดข้อผิดพลาดในการวิเคราะห์");
    }
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    showError(err.message || "เชื่อมต่อเซิร์ฟเวอร์ไม่ได้ ลองใหม่อีกครั้ง");
  } finally {
    loading.hidden = true;
    analyzeBtn.disabled = false;
  }
});

const FACE_ICONS = {
  happy: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="9.2"/>
    <circle cx="8.7" cy="10.2" r="1" fill="currentColor" stroke="none"/>
    <circle cx="15.3" cy="10.2" r="1" fill="currentColor" stroke="none"/>
    <path d="M7.8 14.2c1 1.6 2.6 2.5 4.2 2.5s3.2-.9 4.2-2.5"/>
  </svg>`,
  angry: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="9.2"/>
    <circle cx="8.7" cy="10.6" r="1" fill="currentColor" stroke="none"/>
    <circle cx="15.3" cy="10.6" r="1" fill="currentColor" stroke="none"/>
    <path d="M7 8.6l3 1.4M17 8.6l-3 1.4"/>
    <path d="M8 16.6c1-1.2 2.4-1.8 4-1.8s3 .6 4 1.8"/>
  </svg>`,
};

function renderResult(data) {
  const iconEl = document.getElementById("resultEmoji");
  iconEl.className = "result-icon " + (data.label === "happy" ? "is-happy" : "is-angry");
  iconEl.innerHTML = FACE_ICONS[data.label] || "";
  document.getElementById("resultLabel").textContent = data.label;
  document.getElementById("resultMessage").textContent = data.message;

  const happyPct = Math.round((data.probabilities.happy || 0) * 100);
  const angryPct = Math.round((data.probabilities.angry || 0) * 100);

  // pick mascot: tie (50/50) -> normal, otherwise follow the winning emotion
  setMascot(happyPct === angryPct ? "normal" : data.label);

  document.getElementById("pctHappy").textContent = happyPct + "%";
  document.getElementById("pctAngry").textContent = angryPct + "%";

  const tipsList = document.getElementById("tipsList");
  tipsList.innerHTML = "";
  data.tips.forEach((tip) => {
    const li = document.createElement("li");
    li.textContent = tip;
    tipsList.appendChild(li);
  });

  result.hidden = false;

  requestAnimationFrame(() => {
    document.getElementById("barHappy").style.width = happyPct + "%";
    document.getElementById("barAngry").style.width = angryPct + "%";
  });
}
