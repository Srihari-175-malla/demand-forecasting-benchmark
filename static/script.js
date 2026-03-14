let currentForecastData = null;
let currentStationarityData = null;

document.addEventListener('DOMContentLoaded', () => {
  runForecastComparison();
  fetchStationarity();
});

function toggleTheme() {
  document.body.classList.toggle('light-theme');
  if (currentForecastData) drawForecastCanvas(currentForecastData);
}

function switchNavTab(tabId) {
  document.querySelectorAll('.view-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));

  document.getElementById(tabId).classList.add('active');

  if (tabId === 'charts-view') document.getElementById('tab-btn-charts').classList.add('active');
  if (tabId === 'metrics-view') document.getElementById('tab-btn-metrics').classList.add('active');
  if (tabId === 'stationarity-view') document.getElementById('tab-btn-stationarity').classList.add('active');
  if (tabId === 'analysis-view') document.getElementById('tab-btn-analysis').classList.add('active');
}

async function runForecastComparison() {
  const horizon = parseInt(document.getElementById('horizon-select').value);
  const btn = document.getElementById('solve-btn');

  btn.disabled = true;
  btn.innerText = '⚡ Fitting SARIMA, Prophet & PyTorch LSTM...';

  try {
    const resp = await fetch('/api/forecast', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ horizon: horizon })
    });

    const data = await resp.json();
    currentForecastData = data;

    renderMetricsTable(data.evaluation_rankings);
    renderAnalysisCards(data.tradeoff_analysis.analysis);
    drawForecastCanvas(data);
  } catch (e) {
    console.error('Forecast error:', e);
  } finally {
    btn.disabled = false;
    btn.innerText = '⚡ Run Multi-Model Forecast Comparison';
  }
}

function renderMetricsTable(rankings) {
  const tbody = document.querySelector('#metrics-table tbody');
  if (!rankings) return;

  // Sort by RMSE ascending
  const sorted = [...rankings].sort((a, b) => a.rmse - b.rmse);

  tbody.innerHTML = sorted.map((r, idx) => `
    <tr>
      <td><strong>${r.model_name}</strong></td>
      <td><code>${r.mae} kW</code></td>
      <td><code>${r.rmse} kW</code></td>
      <td><code>${r.mape}%</code></td>
      <td><span style="background:${idx === 0 ? 'rgba(16,185,129,0.2)' : 'rgba(99,102,241,0.2)'}; color:${idx === 0 ? '#10b981' : '#6366f1'}; padding:0.25rem 0.6rem; border-radius:6px; font-weight:700;">Rank #${idx + 1} ${idx === 0 ? '🏆 WINNER' : ''}</span></td>
    </tr>
  `).join('');
}

function renderAnalysisCards(cards) {
  const container = document.getElementById('analysis-container');
  if (!cards) return;

  container.innerHTML = cards.map(c => `
    <div class="analysis-card">
      <h3 style="color:var(--accent-color);">${c.category}</h3>
      <div style="margin-top: 0.5rem; font-size: 0.88rem;">
        <div><strong>Short-Term (7D):</strong> ${c.short_term_perf}</div>
        <div><strong>Long-Term (30D):</strong> ${c.long_term_perf}</div>
      </div>
      <p style="margin-top: 0.75rem; font-size: 0.9rem; color: var(--text-secondary);">${c.tradeoff_explanation}</p>
    </div>
  `).join('');
}

function drawForecastCanvas(data) {
  const canvas = document.getElementById('forecast-canvas');
  if (!canvas || !data) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const isDark = !document.body.classList.contains('light-theme');
  const padding = 50, W = canvas.width, H = canvas.height;

  const yTrue = data.y_true;
  const sarimaF = data.models.sarima.forecasts;
  const prophetF = data.models.prophet.forecasts;
  const lstmF = data.models.lstm.forecasts;

  const allVals = [...yTrue, ...sarimaF, ...prophetF, ...lstmF];
  const minV = Math.min(...allVals) * 0.95;
  const maxV = Math.max(...allVals) * 1.05;

  const scaleX = (i) => padding + (i / (yTrue.length - 1)) * (W - 2 * padding);
  const scaleY = (v) => H - (padding + ((v - minV) / (maxV - minV)) * (H - 2 * padding));

  // Grid
  ctx.strokeStyle = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
  for (let i = 0; i <= 5; i++) {
    const y = padding + i * (H - 2 * padding) / 5;
    ctx.beginPath(); ctx.moveTo(padding, y); ctx.lineTo(W - padding, y); ctx.stroke();
  }

  function drawLine(arr, color, isDashed = false) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    if (isDashed) ctx.setLineDash([5, 5]); else ctx.setLineDash([]);
    arr.forEach((v, i) => {
      const x = scaleX(i), y = scaleY(v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Draw 95% Confidence Band (SARIMA)
  const lowerS = data.models.sarima.lower_bounds;
  const upperS = data.models.sarima.upper_bounds;
  ctx.fillStyle = isDark ? 'rgba(99, 102, 241, 0.12)' : 'rgba(99, 102, 241, 0.15)';
  ctx.beginPath();
  lowerS.forEach((v, i) => {
    const x = scaleX(i), y = scaleY(v);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  for (let i = upperS.length - 1; i >= 0; i--) {
    ctx.lineTo(scaleX(i), scaleY(upperS[i]));
  }
  ctx.closePath();
  ctx.fill();

  drawLine(yTrue, isDark ? '#f8fafc' : '#0f172a');     // Actual True Ground Truth (White/Black)
  drawLine(sarimaF, '#6366f1');                         // SARIMA (Purple)
  drawLine(prophetF, '#10b981');                        // Prophet (Green)
  drawLine(lstmF, '#f59e0b', true);                     // PyTorch LSTM (Orange Dashed)
}

async function fetchStationarity() {
  try {
    const resp = await fetch('/api/stationarity');
    const data = await resp.json();
    currentStationarityData = data;

    const adf = data.adf_test;
    document.getElementById('stat-adf-stat').innerText = adf.adf_statistic;
    document.getElementById('stat-adf-p').innerText = adf.p_value;
    document.getElementById('stat-differencing').innerText = `d = ${adf.recommended_differencing_d}`;

    renderAcfTable(data.acf_pacf);
  } catch (e) {
    console.error('Stationarity error:', e);
  }
}

function renderAcfTable(acfData) {
  const tbody = document.querySelector('#acf-table tbody');
  if (!acfData) return;

  const acf = acfData.acf;
  const pacf = acfData.pacf;

  tbody.innerHTML = acf.map((v, i) => `
    <tr>
      <td>Day ${i+1}</td>
      <td><code>${v}</code></td>
      <td><code>${pacf[i]}</code></td>
    </tr>
  `).join('');
}
