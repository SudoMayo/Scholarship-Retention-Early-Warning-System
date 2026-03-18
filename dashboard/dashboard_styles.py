# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# dashboard_styles.py — Exhibition-Grade CSS & HTML Templates
# Theme: Deep Space Aurora  |  Cyan #06b6d4 + Violet #8b5cf6
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GLOBAL_CSS = """
<style>
/* ── FONTS ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500&display=swap');

/* ── GLOBAL RESET ── */
*, *::before, *::after { box-sizing: border-box; }

/* ── APP BACKGROUND ── */
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 50%, #0a1628 100%) !important;
    background-attachment: fixed !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── HIDE STREAMLIT DEFAULT CHROME ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── REMOVE DEFAULT PADDING ── */
.block-container {
    padding-top: 0rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

/* ── GLASSMORPHISM CARD ── */
.glass-card {
    background: rgba(255, 255, 255, 0.04) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 18px !important;
    padding: 1.5rem !important;
    box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.4),
        0 0 0 1px rgba(255, 255, 255, 0.04),
        inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
    transition: transform 0.3s cubic-bezier(.25,.8,.25,1),
                box-shadow 0.3s ease !important;
    margin-bottom: 1rem;
}
.glass-card:hover {
    transform: translateY(-5px) !important;
    box-shadow:
        0 20px 60px rgba(0, 0, 0, 0.5),
        0 0 40px rgba(6, 182, 212, 0.15) !important;
}

/* ── GRADIENT TEXT ── */
.gradient-text {
    background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'Orbitron', monospace !important;
    font-weight: 900 !important;
    letter-spacing: -0.02em;
    line-height: 1.1;
}

/* ── KPI STYLES ── */
.kpi-number {
    font-family: 'Orbitron', monospace !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #06b6d4, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    display: block;
    margin: 0.4rem 0;
}
.kpi-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: rgba(255,255,255,0.45);
    font-weight: 600;
}

/* ── ANIMATED HERO ── */
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero-section {
    background: linear-gradient(-45deg, #0a0a1a, #0d1b2a, #150a2e, #0a1628);
    background-size: 400% 400%;
    animation: gradientShift 10s ease infinite;
    padding: 3.5rem 2rem 2.5rem 2rem;
    border-radius: 0 0 24px 24px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

/* ── ENTRANCE ANIMATIONS ── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(28px); }
    to   { opacity: 1; transform: translateY(0); }
}
.animate-1 { animation: fadeSlideUp 0.6s ease 0.1s both; }
.animate-2 { animation: fadeSlideUp 0.6s ease 0.2s both; }
.animate-3 { animation: fadeSlideUp 0.6s ease 0.3s both; }
.animate-4 { animation: fadeSlideUp 0.6s ease 0.4s both; }
.animate-5 { animation: fadeSlideUp 0.6s ease 0.5s both; }
.animate-6 { animation: fadeSlideUp 0.6s ease 0.6s both; }

/* ── SHIMMER BADGE ── */
@keyframes shimmer {
    0%   { background-position: -400% 0; }
    100% { background-position: 400% 0; }
}
.shimmer-badge {
    background: linear-gradient(90deg,
        rgba(255,255,255,0.06) 25%,
        rgba(255,255,255,0.14) 50%,
        rgba(255,255,255,0.06) 75%);
    background-size: 400% 100%;
    animation: shimmer 3s infinite;
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 999px;
    padding: 0.3rem 1rem;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.75);
    display: inline-block;
    margin: 0 0.25rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
}

/* ── LIVE PULSE ── */
@keyframes pulseRing {
    0%   { transform: scale(0.8); opacity: 1; }
    100% { transform: scale(2.0); opacity: 0; }
}
.live-container {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 999px;
    padding: 0.25rem 0.75rem 0.25rem 0.5rem;
    font-size: 0.78rem;
    color: #22c55e;
    font-weight: 600;
}
.live-dot {
    width: 8px; height: 8px;
    background: #22c55e;
    border-radius: 50%;
    position: relative;
    flex-shrink: 0;
}
.live-dot::after {
    content: '';
    position: absolute;
    inset: -3px;
    border-radius: 50%;
    background: #22c55e;
    animation: pulseRing 1.8s ease-out infinite;
}

/* ── METRIC OVERRIDES ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    padding: 1.2rem 1.5rem !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.4), 0 0 20px rgba(6,182,212,0.15) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: rgba(255,255,255,0.5) !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
}

/* ── TAB OVERRIDES ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px !important;
    padding: 0.3rem !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    gap: 0.3rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: rgba(255,255,255,0.5) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: all 0.25s ease !important;
    border: none !important;
    padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(6,182,212,0.3), rgba(139,92,246,0.3)) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 12px rgba(6,182,212,0.25) !important;
}

/* ── BUTTON OVERRIDES ── */
.stButton > button {
    background: linear-gradient(135deg, #06b6d4, #8b5cf6) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.6rem 1.5rem !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(6,182,212,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 30px rgba(6,182,212,0.5) !important;
}

/* ── SELECT / SLIDER / INPUT ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stNumberInput input, .stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    overflow: hidden !important;
}

/* ── SECTION DIVIDER ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #06b6d4, transparent);
    margin: 2.5rem 0;
    opacity: 0.4;
}

/* ── PIPELINE STEP ── */
.pipeline-step {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    position: relative;
    transition: all 0.3s ease;
}
.pipeline-step:hover {
    background: rgba(255,255,255,0.07);
    border-color: rgba(6,182,212,0.4);
    transform: translateY(-4px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
.pipeline-step-icon { font-size: 2rem; margin-bottom: 0.6rem; }
.pipeline-step-number {
    position: absolute;
    top: -10px; left: -10px;
    width: 24px; height: 24px;
    background: linear-gradient(135deg, #06b6d4, #8b5cf6);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 800;
    color: white;
    font-family: 'Orbitron', monospace;
}

/* ── ARROW CONNECTOR ── */
@keyframes arrowPulse {
    0%, 100% { opacity: 0.4; transform: translateX(0); }
    50%       { opacity: 1.0; transform: translateX(4px); }
}
.arrow-connector {
    display: flex;
    align-items: center;
    justify-content: center;
    color: rgba(6,182,212,0.6);
    font-size: 1.5rem;
    animation: arrowPulse 2s ease-in-out infinite;
}

/* ── TECH BADGE ── */
.tech-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 0.5rem 0.9rem;
    font-size: 0.82rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    color: rgba(255,255,255,0.8);
    transition: all 0.25s ease;
    cursor: default;
    margin: 0.25rem;
}
.tech-badge:hover {
    background: rgba(255,255,255,0.1);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}
.tech-badge.ml       { border-left: 3px solid #8b5cf6; }
.tech-badge.data     { border-left: 3px solid #06b6d4; }
.tech-badge.backend  { border-left: 3px solid #10b981; }
.tech-badge.viz      { border-left: 3px solid #f59e0b; }
.tech-badge.db       { border-left: 3px solid #ef4444; }

/* ── INSIGHT CARD ── */
.insight-card {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid #06b6d4;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    transition: all 0.25s ease;
}
.insight-card:hover {
    background: rgba(255,255,255,0.06);
    border-left-color: #8b5cf6;
    transform: translateX(4px);
}

/* ── ALERT CARDS ── */
.alert-critical {
    background: linear-gradient(135deg, #2d1f2f 0%, #3d1f1f 100%);
    border-left: 4px solid #ff4757;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}
.alert-warning-card {
    background: linear-gradient(135deg, #2d2a1f 0%, #3d351f 100%);
    border-left: 4px solid #ffa502;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}
.alert-safe {
    background: linear-gradient(135deg, #1f2d22 0%, #1f3d24 100%);
    border-left: 4px solid #2ed573;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}

/* ── FOOTER ── */
.dashboard-footer {
    border-top: 1px solid rgba(255,255,255,0.06);
    padding: 1.5rem 0 0.5rem 0;
    text-align: center;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.3);
    margin-top: 3rem;
}
.footer-gradient-line {
    height: 2px;
    background: linear-gradient(90deg, transparent, #06b6d4, #8b5cf6, transparent);
    margin-bottom: 1.2rem;
    opacity: 0.5;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(#06b6d4, #8b5cf6);
    border-radius: 3px;
}
</style>
"""

# ── HERO HTML ──────────────────────────────────────────────
HERO_HTML = """
<div class="hero-section animate-1">
    <div style="text-align:center; position:relative; z-index:2;">
        <div style="margin-bottom:1rem;">
            <span class="live-container">
                <span class="live-dot"></span> LIVE DEMO READY
            </span>
        </div>
        <h1 class="gradient-text animate-2"
            style="font-size:clamp(2.2rem,5vw,4rem);
                   margin-bottom:0.5rem; text-shadow: 0 0 60px rgba(6,182,212,0.3);">
            SCHOLARGUARD
        </h1>
        <p class="animate-3"
           style="font-size:1.15rem; color:rgba(255,255,255,0.65);
                  max-width:700px; margin:0 auto 1.5rem;
                  font-family:'Space Grotesk'; font-weight:400; line-height:1.6;">
            Scholarship Retention Early Warning System — predicting academic risk
            before it's too late, enabling early intervention at Vijaybhoomi University
        </p>
        <div class="animate-4" style="margin-bottom:2rem;">
            <span class="shimmer-badge">🐍 Python 3.11</span>
            <span class="shimmer-badge">🤖 XGBoost + Random Forest</span>
            <span class="shimmer-badge">📊 Multi-Semester Analytics</span>
            <span class="shimmer-badge">🗄️ SQLite Pipeline</span>
            <span class="shimmer-badge"
                  style="border-color:rgba(34,197,94,0.4);
                         color:#22c55e;">✅ EXHIBITION 2026</span>
        </div>
        <div class="animate-5"
             style="display:flex; justify-content:center; gap:3rem; flex-wrap:wrap;">
            <div style="text-align:center;">
                <div class="kpi-number" style="font-size:2rem;">97.7%</div>
                <div class="kpi-label">RISK RECALL</div>
            </div>
            <div style="text-align:center; border-left:1px solid rgba(255,255,255,0.1);
                        padding-left:3rem;">
                <div class="kpi-number" style="font-size:2rem;">19,200</div>
                <div class="kpi-label">ACADEMIC RECORDS</div>
            </div>
            <div style="text-align:center; border-left:1px solid rgba(255,255,255,0.1);
                        padding-left:3rem;">
                <div class="kpi-number" style="font-size:2rem;">3</div>
                <div class="kpi-label">ML MODELS TRAINED</div>
            </div>
        </div>
    </div>
</div>
"""

# ── PARTICLE JS ───────────────────────────────────────────
PARTICLE_JS = """
<canvas id="c" style="position:fixed;top:0;left:0;
    width:100vw;height:100vh;pointer-events:none;z-index:0;opacity:0.3;"></canvas>
<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
const particles = Array.from({length: 50}, () => ({
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    r: Math.random() * 2 + 0.5,
    dx: (Math.random() - 0.5) * 0.3,
    dy: (Math.random() - 0.5) * 0.3,
    alpha: Math.random() * 0.5 + 0.1
}));
function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach((p, i) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = i % 2 === 0
            ? `rgba(6,182,212,${p.alpha})`
            : `rgba(139,92,246,${p.alpha})`;
        ctx.fill();
        p.x += p.dx; p.y += p.dy;
        if (p.x < 0 || p.x > canvas.width) p.dx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.dy *= -1;
    });
    // Draw connections
    for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
            const dx = particles[i].x - particles[j].x;
            const dy = particles[i].y - particles[j].y;
            const dist = Math.sqrt(dx*dx + dy*dy);
            if (dist < 120) {
                ctx.beginPath();
                ctx.moveTo(particles[i].x, particles[i].y);
                ctx.lineTo(particles[j].x, particles[j].y);
                ctx.strokeStyle = `rgba(6,182,212,${0.08 * (1 - dist/120)})`;
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }
        }
    }
    requestAnimationFrame(draw);
}
draw();
window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
});
</script>
"""

# ── COUNTER JS ────────────────────────────────────────────
COUNTER_JS = """
<script>
setTimeout(function() {
    document.querySelectorAll('[data-testid="stMetricValue"]').forEach(function(el) {
        const text = el.textContent;
        const target = parseFloat(text.replace(/[^0-9.]/g, ''));
        if (isNaN(target)) return;
        const suffix = text.replace(/[0-9.,]/g, '');
        const hasDecimal = text.includes('.');
        const decimals = hasDecimal ? (text.split('.')[1] || '').replace(/[^0-9]/g, '').length : 0;
        let startTime = performance.now();
        function update(now) {
            let p = Math.min((now - startTime) / 1600, 1);
            let ease = 1 - Math.pow(1 - p, 3);
            let val = (target * ease).toFixed(decimals);
            if (text.includes(',')) {
                val = Number(val).toLocaleString('en-US', {minimumFractionDigits: decimals});
            }
            el.textContent = val + suffix;
            if (p < 1) requestAnimationFrame(update);
        }
        requestAnimationFrame(update);
    });
}, 400);
</script>
"""

# ── PIPELINE HTML ─────────────────────────────────────────
PIPELINE_HTML = """
<div style="display:grid; grid-template-columns:1fr auto 1fr auto 1fr auto 1fr; align-items:center; gap:0.5rem; margin:1.5rem 0;">
    <div class="pipeline-step">
        <div class="pipeline-step-number">1</div>
        <div class="pipeline-step-icon">📥</div>
        <div style="font-weight:600; color:#e2e8f0; font-size:0.9rem;">Data Ingestion</div>
        <div style="font-size:0.75rem; color:rgba(255,255,255,0.45); margin-top:0.3rem;">
            800 students × 4 semesters<br>SQLite + CSV pipeline
        </div>
    </div>
    <div class="arrow-connector">→</div>
    <div class="pipeline-step">
        <div class="pipeline-step-number">2</div>
        <div class="pipeline-step-icon">⚙️</div>
        <div style="font-weight:600; color:#e2e8f0; font-size:0.9rem;">Feature Engineering</div>
        <div style="font-size:0.75rem; color:rgba(255,255,255,0.45); margin-top:0.3rem;">
            10 features extracted<br>OneHot + Numeric pipeline
        </div>
    </div>
    <div class="arrow-connector">→</div>
    <div class="pipeline-step">
        <div class="pipeline-step-number">3</div>
        <div class="pipeline-step-icon">🤖</div>
        <div style="font-weight:600; color:#e2e8f0; font-size:0.9rem;">ML Prediction</div>
        <div style="font-size:0.75rem; color:rgba(255,255,255,0.45); margin-top:0.3rem;">
            XGBoost classifier<br>9-class grade prediction
        </div>
    </div>
    <div class="arrow-connector">→</div>
    <div class="pipeline-step">
        <div class="pipeline-step-number">4</div>
        <div class="pipeline-step-icon">🎯</div>
        <div style="font-weight:600; color:#e2e8f0; font-size:0.9rem;">Risk Assessment</div>
        <div style="font-size:0.75rem; color:rgba(255,255,255,0.45); margin-top:0.3rem;">
            CGPA projection<br>Scholarship risk scoring
        </div>
    </div>
</div>
"""

# ── TECH STACK HTML ───────────────────────────────────────
TECH_STACK_HTML = """
<div style="display:flex; flex-wrap:wrap; gap:0.4rem; justify-content:center; margin:1rem 0;">
    <span class="tech-badge ml">🤖 scikit-learn</span>
    <span class="tech-badge ml">🌲 XGBoost</span>
    <span class="tech-badge ml">📊 Logistic Regression</span>
    <span class="tech-badge ml">🌳 Random Forest</span>
    <span class="tech-badge data">🐼 pandas</span>
    <span class="tech-badge data">🔢 NumPy</span>
    <span class="tech-badge viz">📈 Plotly</span>
    <span class="tech-badge viz">🖥️ Streamlit</span>
    <span class="tech-badge db">🗄️ SQLite</span>
    <span class="tech-badge backend">🐍 Python 3.11</span>
    <span class="tech-badge backend">💾 Joblib</span>
</div>
"""
