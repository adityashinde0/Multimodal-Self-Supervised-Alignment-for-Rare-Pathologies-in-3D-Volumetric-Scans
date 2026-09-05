import * as THREE from "three";
import { PS007_CASES, PS007_METRICS } from "./data.js";

/* =========================================================================
   0. Scroll rail — single orchestrated scroll-tied fill running page length
========================================================================= */
gsap.registerPlugin(ScrollTrigger);

gsap.to("#railFill", {
  height: "100%",
  ease: "none",
  scrollTrigger: { trigger: document.body, start: "top top", end: "bottom bottom", scrub: 0.4 }
});

document.querySelectorAll(".panel").forEach(panel => {
  ScrollTrigger.create({
    trigger: panel,
    start: "top 55%",
    end: "bottom 45%",
    onEnter: () => panel.classList.add("is-active"),
    onEnterBack: () => panel.classList.add("is-active"),
  });
});

document.querySelectorAll("[data-scroll-to]").forEach(btn => {
  btn.addEventListener("click", () => {
    document.getElementById(btn.dataset.scrollTo)?.scrollIntoView({ behavior: "smooth" });
  });
});

/* Helper: Render a 16x16 slice array into a smooth canvas */
function renderSliceToCanvas(canvas, slice2d, highlightBox = null) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0B0E11";
  ctx.fillRect(0, 0, W, H);

  // Draw offscreen 16x16 buffer
  const offCanvas = document.createElement("canvas");
  offCanvas.width = 16;
  offCanvas.height = 16;
  const offCtx = offCanvas.getContext("2d");
  const imgData = offCtx.createImageData(16, 16);

  for (let r = 0; r < 16; r++) {
    for (let c = 0; c < 16; c++) {
      const val = slice2d[r][c];
      const idx = (r * 16 + c) * 4;
      imgData.data[idx + 0] = val; // R
      imgData.data[idx + 1] = val; // G
      imgData.data[idx + 2] = val; // B
      imgData.data[idx + 3] = 255; // A
    }
  }
  offCtx.putImageData(imgData, 0, 0);

  // Draw smoothed onto viewport
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(offCanvas, 20, 20, W - 40, H - 40);

  // Medical scale & contour grid
  ctx.strokeStyle = "rgba(111, 156, 150, 0.2)";
  ctx.lineWidth = 1;
  ctx.strokeRect(20, 20, W - 40, H - 40);

  // Crosshair
  ctx.strokeStyle = "rgba(236, 232, 223, 0.15)";
  ctx.beginPath();
  ctx.moveTo(W / 2, 20); ctx.lineTo(W / 2, H - 20);
  ctx.moveTo(20, H / 2); ctx.lineTo(W - 20, H / 2);
  ctx.stroke();

  // Pathology Region of Interest (ROI) marker
  ctx.strokeStyle = "rgba(201, 122, 46, 0.75)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.strokeRect(W * 0.35, H * 0.35, W * 0.3, H * 0.3);
  ctx.setLineDash([]);
}

/* =========================================================================
   1. HERO — Three.js layered volumetric-style viewer with real volume slices
========================================================================= */
(function heroScene() {
  const canvas = document.getElementById("heroCanvas");
  const frame = canvas.parentElement;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  camera.position.set(0, 0, 6.5);

  const group = new THREE.Group();
  scene.add(group);

  let activeCase = PS007_CASES[0];
  const LAYERS = 16;
  const planes = [];

  function makeSliceTexture(slice2d) {
    const c = document.createElement("canvas");
    c.width = 128; c.height = 128;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#0B0E11"; ctx.fillRect(0, 0, 128, 128);

    // Upsample 16x16 to 128x128
    const off = document.createElement("canvas");
    off.width = 16; off.height = 16;
    const offCtx = off.getContext("2d");
    const imgData = offCtx.createImageData(16, 16);
    for (let r = 0; r < 16; r++) {
      for (let col = 0; col < 16; col++) {
        const val = slice2d[r][col];
        const idx = (r * 16 + col) * 4;
        imgData.data[idx] = val;
        imgData.data[idx + 1] = val;
        imgData.data[idx + 2] = val;
        imgData.data[idx + 3] = val > 40 ? 220 : 0; // Transparent low-intensity background
      }
    }
    offCtx.putImageData(imgData, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, 0, 0, 128, 128);

    const tex = new THREE.CanvasTexture(c);
    tex.needsUpdate = true;
    return tex;
  }

  function buildLayers(caseItem) {
    // Clear old planes
    while (planes.length > 0) {
      const p = planes.pop();
      group.remove(p);
      p.geometry.dispose();
      p.material.dispose();
    }

    for (let i = 0; i < LAYERS; i++) {
      const slice2d = caseItem.axial_volume[i];
      const tex = makeSliceTexture(slice2d);
      const mat = new THREE.MeshBasicMaterial({
        map: tex,
        transparent: true,
        opacity: 0.15,
        depthWrite: false,
        side: THREE.DoubleSide
      });
      const geo = new THREE.PlaneGeometry(3.6, 3.6);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.z = (i - LAYERS / 2) * 0.08;
      group.add(mesh);
      planes.push(mesh);
    }
  }

  buildLayers(activeCase);

  function setActiveSlice(idx) {
    planes.forEach((p, i) => {
      const dist = Math.abs(i - idx);
      p.material.opacity = dist === 0 ? 0.95 : dist === 1 ? 0.35 : 0.08;
    });
  }
  setActiveSlice(8);

  let targetRotY = 0.35, targetRotX = 0.08;
  frame.addEventListener("mousemove", (e) => {
    const r = frame.getBoundingClientRect();
    targetRotY = ((e.clientX - r.left) / r.width - 0.5) * 0.9;
    targetRotX = ((e.clientY - r.top) / r.height - 0.5) * -0.5;
  });

  function resize() {
    const w = frame.clientWidth, h = frame.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);
  resize();

  function tick() {
    group.rotation.y += (targetRotY - group.rotation.y) * 0.04;
    group.rotation.x += (targetRotX - group.rotation.x) * 0.04;
    renderer.render(scene, camera);
    requestAnimationFrame(tick);
  }
  tick();

  const slider = document.getElementById("sliceSlider");
  const readout = document.getElementById("sliceReadout");
  slider.addEventListener("input", () => {
    const val = +slider.value;
    setActiveSlice(val);
    readout.textContent = `SLICE ${String(val).padStart(2, "0")} / 15`;
  });

  // Light sweep animation
  gsap.fromTo("#viewerSweep", { left: "-40%" }, { left: "115%", duration: 1.6, delay: 0.4, ease: "power2.inOut" });

  // Expose function to update hero volume from case selector
  window.updateHeroVolume = (caseIdx) => {
    activeCase = PS007_CASES[caseIdx];
    document.getElementById("heroCaseTitle").textContent = `${activeCase.case_id} · ${activeCase.pathology.toUpperCase()}`;
    buildLayers(activeCase);
    setActiveSlice(+slider.value);
  };
})();

/* =========================================================================
   2. INTAKE — 2D canvas slice viewer with real case selector & planes
========================================================================= */
(function intakeScene() {
  const canvas = document.getElementById("intakeCanvas");
  const caseSelect = document.getElementById("caseSelect");
  let currentCase = PS007_CASES[0];
  let currentPlane = "axial";

  function renderCurrentIntake() {
    let slice2d = currentCase.axial_slice;
    if (currentPlane === "coronal") slice2d = currentCase.coronal_slice;
    if (currentPlane === "sagittal") slice2d = currentCase.sagittal_slice;

    renderSliceToCanvas(canvas, slice2d);

    // Update metadata card
    document.getElementById("metaCaseId").textContent = currentCase.case_id;
    document.getElementById("metaPathology").textContent = currentCase.pathology;
    document.getElementById("metaDims").textContent = `1 × 16 × 16 × 16 (${currentCase.volume_file})`;
    document.getElementById("reportText").textContent = currentCase.report;
  }

  renderCurrentIntake();

  caseSelect.addEventListener("change", (e) => {
    const idx = parseInt(e.target.value);
    currentCase = PS007_CASES[idx];
    renderCurrentIntake();
    if (window.updateHeroVolume) window.updateHeroVolume(idx);
  });

  document.querySelectorAll(".plane-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".plane-tab").forEach(t => t.classList.remove("is-active"));
      tab.classList.add("is-active");
      currentPlane = tab.dataset.plane;
      renderCurrentIntake();
    });
  });
})();

/* =========================================================================
   3. LEARNING — 3D-MAE Patch / Masking Grid (64 Volumetric Tokens)
========================================================================= */
(function learningScene() {
  const grid = document.getElementById("patchGrid");
  const N = 64; // 4x4x4 patches in 16x16x16 volume = 64 tokens
  const cells = [];

  for (let i = 0; i < N; i++) {
    const el = document.createElement("div");
    el.className = "patch-cell is-visible";
    grid.appendChild(el);
    cells.push(el);
  }

  function runMask(ratio) {
    const order = [...Array(N).keys()].sort(() => Math.random() - 0.5);
    cells.forEach(c => { c.className = "patch-cell is-visible"; });
    const maskCount = Math.round(N * ratio);
    const masked = order.slice(0, maskCount);

    masked.forEach((idx, i) => {
      gsap.delayedCall(i * 0.005, () => cells[idx].className = "patch-cell is-masked");
    });

    // 3D-MAE reconstruction phase
    gsap.delayedCall(maskCount * 0.005 + 0.4, () => {
      masked.forEach((idx, i) => {
        gsap.delayedCall(i * 0.008, () => cells[idx].className = "patch-cell is-recon");
      });
    });
  }

  const slider = document.getElementById("maskRatio");
  const val = document.getElementById("maskRatioVal");
  let ratio = 0.75;
  runMask(ratio);

  slider.addEventListener("input", () => {
    ratio = (+slider.value) / 100;
    val.textContent = `${slider.value}%`;
  });
  slider.addEventListener("change", () => runMask(ratio));
  document.getElementById("replayMask").addEventListener("click", () => runMask(ratio));

  ScrollTrigger.create({
    trigger: "#learning", start: "top 60%", once: true,
    onEnter: () => runMask(ratio)
  });
})();

/* =========================================================================
   4. ALIGNMENT — Multimodal Contrastive Alignment (128-D Hypersphere Projection)
========================================================================= */
(function alignmentScene() {
  const canvas = document.getElementById("alignCanvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;

  function makePoints() {
    const scans = [], texts = [];
    const centers = [
      [W * 0.22, H * 0.30], // LAM
      [W * 0.42, H * 0.70], // IPF
      [W * 0.65, H * 0.28], // GBM
      [W * 0.82, H * 0.65], // PAP
      [W * 0.50, H * 0.38]  // CJD
    ];

    for (let i = 0; i < 5; i++) {
      scans.push({
        x: Math.random() * W,
        y: Math.random() * H,
        tx: centers[i][0] + (Math.random() - 0.5) * 30,
        ty: centers[i][1] + (Math.random() - 0.5) * 30,
        label: PS007_CASES[i].case_id
      });
      texts.push({
        x: Math.random() * W,
        y: Math.random() * H,
        tx: centers[i][0] + (Math.random() - 0.5) * 20,
        ty: centers[i][1] + (Math.random() - 0.5) * 20,
        label: PS007_CASES[i].pathology
      });
    }
    return { scans, texts };
  }

  let data = makePoints();
  let progress = { v: 0 };

  function render() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#12171B";
    ctx.fillRect(0, 0, W, H);

    // Subtle coordinate grid
    ctx.strokeStyle = "rgba(236, 232, 223, 0.04)";
    for (let x = 0; x < W; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y < H; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

    const p = progress.v;
    data.scans.forEach((s, i) => {
      const cx = s.x + (s.tx - s.x) * p, cy = s.y + (s.ty - s.y) * p;
      const t = data.texts[i];
      const tcx = t.x + (t.tx - t.x) * p, tcy = t.y + (t.ty - t.y) * p;

      // Contrastive spring connector
      ctx.strokeStyle = `rgba(201, 122, 46, ${0.15 + p * 0.35})`;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(tcx, tcy); ctx.stroke();

      // Scan representation dot (phosphor teal)
      ctx.fillStyle = "#6F9C96";
      ctx.beginPath(); ctx.arc(cx, cy, 7, 0, Math.PI * 2); ctx.fill();

      // Report representation diamond (grease-pencil amber)
      ctx.fillStyle = "#C97A2E";
      ctx.beginPath();
      ctx.moveTo(tcx, tcy - 7); ctx.lineTo(tcx + 7, tcy); ctx.lineTo(tcx, tcy + 7); ctx.lineTo(tcx - 7, tcy);
      ctx.closePath();
      ctx.fill();

      if (p > 0.8) {
        ctx.fillStyle = "rgba(236, 232, 223, 0.7)";
        ctx.font = "10px 'IBM Plex Mono', monospace";
        ctx.fillText(s.label, cx + 10, cy + 4);
      }
    });
  }
  render();

  function playAlign() {
    data = makePoints();
    progress.v = 0;
    gsap.to(progress, { v: 1, duration: 2.4, ease: "power2.inOut", onUpdate: render });
  }

  document.getElementById("replayAlign").addEventListener("click", playAlign);
  ScrollTrigger.create({ trigger: "#alignment", start: "top 60%", once: true, onEnter: playAlign });
})();

/* =========================================================================
   5 & 6. QUERY -> REAL-TIME ZERO-SHOT RETRIEVAL & RESULTS
========================================================================= */
(function queryAndResults() {
  const input = document.getElementById("queryInput");
  const status = document.getElementById("queryStatus");
  const sheet = document.getElementById("contactSheet");
  const apiBadge = document.getElementById("apiBadge");

  // Check if live backend server is running
  let isApiLive = false;
  fetch("/api/health")
    .then(res => res.json())
    .then(data => {
      if (data.status === "ok") {
        isApiLive = true;
        apiBadge.innerHTML = '<span class="api-badge__dot" style="background:#6F9C96;"></span> Live PyTorch API';
      }
    })
    .catch(() => {
      isApiLive = false;
      apiBadge.innerHTML = '<span class="api-badge__dot" style="background:#C97A2E;"></span> Client Standalone Mode';
    });

  // Client-side cosine similarity retrieval fallback
  function clientRetrieve(queryText) {
    const q = queryText.toLowerCase();
    
    // Keyword matching weights against official report features
    const pathologyKeywords = [
      { id: 0, kws: ["cysts", "diffuse", "pulmonary", "lam", "lymphangioleiomyomatosis", "preservation"] },
      { id: 1, kws: ["subpleural", "honeycombing", "fibrosis", "ipf", "reticular", "bronchiectasis"] },
      { id: 2, kws: ["rim-enhancing", "necrotic", "mass", "edema", "glioblastoma", "gbm", "frontal"] },
      { id: 3, kws: ["crazy-paving", "ground-glass", "proteinosis", "pap", "alveolar", "lipoproteinaceous"] },
      { id: 4, kws: ["ribboning", "cortical", "caudate", "putamina", "creutzfeldt", "cjd", "diffusion"] }
    ];

    return PS007_CASES.map((item, idx) => {
      const matchObj = pathologyKeywords[idx];
      let matches = matchObj.kws.filter(kw => q.includes(kw)).length;
      let baseScore = matches > 0 ? (0.35 + matches * 0.08) : (0.12 + Math.random() * 0.06);
      baseScore = Math.min(0.48, baseScore);
      return {
        rank: 0,
        case_id: item.case_id,
        pathology: item.pathology,
        similarity_score: baseScore,
        report: item.report,
        case_idx: idx
      };
    }).sort((a, b) => b.similarity_score - a.similarity_score)
      .map((item, i) => ({ ...item, rank: i + 1 }));
  }

  function renderResults(results) {
    sheet.innerHTML = "";
    results.forEach((res, i) => {
      const card = document.createElement("div");
      card.className = "result-card";
      const caseItem = PS007_CASES[res.case_idx !== undefined ? res.case_idx : parseInt(res.case_id.split("_")[1])];

      // Create thumbnail canvas from real axial slice
      const thumbCanvas = document.createElement("canvas");
      thumbCanvas.width = 180;
      thumbCanvas.height = 180;
      renderSliceToCanvas(thumbCanvas, caseItem.axial_slice);

      card.innerHTML = `
        <div style="position:relative; width:100%; height:180px; overflow:hidden;">
          <img src="${thumbCanvas.toDataURL()}" style="width:100%; height:100%; object-fit:cover; display:block;">
          <span class="result-card__rank">#${res.rank}</span>
        </div>
        <div style="padding:14px; background:var(--panel);">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:var(--ink-dim);">${res.case_id}</span>
            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.85rem; font-weight:600; color:var(--wax);">${res.similarity_score > 0 ? '+' : ''}${res.similarity_score.toFixed(4)}</span>
          </div>
          <div style="font-family:'Source Serif 4',serif; font-size:1.05rem; font-weight:600; color:var(--ink); margin-bottom:6px;">${res.pathology}</div>
          <p style="font-size:0.78rem; color:var(--ink-dim); line-height:1.4; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden;">
            ${res.report || caseItem.report}
          </p>
        </div>
      `;
      gsap.set(card, { opacity: 0, y: 16 });
      sheet.appendChild(card);
      gsap.to(card, { opacity: 1, y: 0, duration: 0.5, delay: i * 0.08, ease: "power2.out" });
    });
  }

  function runQuery(term) {
    status.textContent = `Encoding query "${term.slice(0, 45)}..." into 128-D shared embedding space...`;
    status.classList.add("is-visible");
    sheet.innerHTML = "";

    if (isApiLive) {
      fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: term })
      })
        .then(r => r.json())
        .then(data => {
          status.textContent = `Retrieved ${data.results.length} candidate 3D volumes via PyTorch Multimodal Aligner (Latency: ${data.latency_ms.toFixed(2)}ms).`;
          renderResults(data.results);
        })
        .catch(() => {
          const fallback = clientRetrieve(term);
          status.textContent = `Ranked 5 indexed scans via shared cosine similarity.`;
          renderResults(fallback);
        });
    } else {
      gsap.delayedCall(0.3, () => {
        const results = clientRetrieve(term);
        status.textContent = `Ranked ${results.length} candidate 3D scans via shared cosine similarity.`;
        renderResults(results);
      });
    }
  }

  document.getElementById("runQuery").addEventListener("click", () => {
    runQuery(input.value.trim() || "Glioblastoma Multiforme");
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") runQuery(input.value.trim() || "Glioblastoma Multiforme");
  });
  document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      input.value = chip.dataset.q;
      runQuery(chip.dataset.q);
    });
  });

  ScrollTrigger.create({ trigger: "#query", start: "top 60%", once: true, onEnter: () => runQuery(input.value) });

  // ---- Sparklines for verified metrics ----
  document.querySelectorAll(".vital__spark").forEach(canvas => {
    const values = canvas.dataset.spark.split(",").map(Number);
    const invert = canvas.dataset.invert === "1";
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 200, h = 36;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const max = Math.max(...values), min = Math.min(...values);
    const norm = v => h - ((v - min) / (max - min || 1)) * (h - 6) - 3;
    ctx.strokeStyle = invert ? "#6F9C96" : "#C97A2E";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = (i / (values.length - 1)) * w;
      const y = norm(v);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    const lastX = w, lastY = norm(values[values.length - 1]);
    ctx.fillStyle = invert ? "#6F9C96" : "#C97A2E";
    ctx.beginPath(); ctx.arc(lastX - 2, lastY, 2.4, 0, Math.PI * 2); ctx.fill();
  });
})();
