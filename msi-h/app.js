function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  if (!lines.length) {
    return [];
  }
  const headers = splitCsvLine(lines[0]);
  return lines.slice(1).filter(Boolean).map((line) => {
    const values = splitCsvLine(line);
    const row = {};
    headers.forEach((header, index) => {
      row[header] = values[index] ?? "";
    });
    return row;
  });
}

function splitCsvLine(line) {
  const out = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === "," && !inQuotes) {
      out.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  out.push(current);
  return out;
}

function normalizeKey(value) {
  return (value || "")
    .trim()
    .toLowerCase()
    .replace(/\.(svs|tif|tiff|png|jpg|jpeg)$/i, "");
}

function fmt(value, digits = 4) {
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(digits) : "-";
}

function setText(id, value) {
  const node = document.getElementById(id);
  if (node) {
    node.textContent = value;
  }
}

async function loadBundle() {
  const [metricsRes, slidesRes] = await Promise.all([
    fetch("./data/metrics.json"),
    fetch("./data/selected_200_slides.csv"),
  ]);
  const metrics = await metricsRes.json();
  const slides = parseCsv(await slidesRes.text());
  return { metrics, slides };
}

function renderSummary(metrics) {
  setText("threshold-value", fmt(metrics.mean_best_threshold));
  setText("auroc-value", fmt(metrics.mean_auroc));
  setText("recall-value", fmt(metrics.mean_recall_msi_h));
  setText("specificity-value", fmt(metrics.mean_specificity));
}

function bindUpload(bundle) {
  const input = document.getElementById("file-input");
  const form = document.getElementById("upload-form");
  const card = document.getElementById("result-card");
  const bySlide = new Map();
  const byPatient = new Map();

  bundle.slides.forEach((row) => {
    bySlide.set(normalizeKey(row.bucket_name || row.slide), row);
    byPatient.set(normalizeKey(row.patient), row);
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const file = input.files && input.files[0];
    if (!file) {
      card.className = "result-card";
      card.innerHTML = `
        <h2>No file selected</h2>
        <p class="result-note">Choose a file first so the app can check its name against the preserved trained cohort.</p>
      `;
      return;
    }

    const key = normalizeKey(file.name);
    const row = bySlide.get(key) || byPatient.get(key);

    if (!row) {
      card.className = "result-card";
      card.innerHTML = `
        <h2>Saved output not found</h2>
        <p class="result-note">
          The uploaded file name does not match a preserved slide from the local trained bundle.
          This app only returns saved trained-output details for known cohort items.
        </p>
        <div class="result-grid">
          <span>Uploaded file</span><strong>${file.name}</strong>
          <span>Model</span><strong>H-Optimus-0</strong>
          <span>Threshold</span><strong>${fmt(bundle.metrics.mean_best_threshold)}</strong>
          <span>Mode</span><strong>Saved bundle only</strong>
        </div>
      `;
      return;
    }

    card.className = "result-card";
    card.innerHTML = `
      <h2>Saved trained output</h2>
      <div class="result-grid">
        <span>Uploaded file</span><strong>${file.name}</strong>
        <span>Matched slide</span><strong>${row.bucket_name || row.slide}</strong>
        <span>Patient</span><strong>${row.patient || "-"}</strong>
        <span>MSI status</span><strong>${row.msi_status || "-"}</strong>
        <span>Repeat</span><strong>${row.repeat || "-"}</strong>
        <span>Fold</span><strong>${row.fold || "-"}</strong>
        <span>Model</span><strong>H-Optimus-0</strong>
        <span>Head</span><strong>TransMIL</strong>
        <span>Mean threshold</span><strong>${fmt(bundle.metrics.mean_best_threshold)}</strong>
        <span>Mean AUROC</span><strong>${fmt(bundle.metrics.mean_auroc)}</strong>
      </div>
      <p class="result-note">
        This is a result from the preserved trained cohort bundle. It is not fresh neural inference on the uploaded file pixels.
      </p>
    `;
  });
}

async function main() {
  const bundle = await loadBundle();
  renderSummary(bundle.metrics);
  bindUpload(bundle);
}

main();
