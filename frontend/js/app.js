// ---------------------------------------------------------
// CUSTOM JAVASCRIPT - AI CUSTOMER FEEDBACK INTELLIGENCE SPA
// ---------------------------------------------------------

const API_BASE = ""; // Relative URL since Flask serves both frontend and API

document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initSampleButtons();
    checkApiHealth();
    loadAnalyticsData();
    
    // Bind Analyze Form
    const analyzeBtn = document.getElementById("analyzeBtn");
    if (analyzeBtn) {
        analyzeBtn.addEventListener("click", analyzeReview);
    }
});

// ---------------------------------------------------------
// 1. SPA NAVIGATION
// ---------------------------------------------------------
function initNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    const pageViews = document.querySelectorAll(".page-view");

    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const targetPage = item.getAttribute("data-page");

            navItems.forEach(i => i.classList.remove("active"));
            pageViews.forEach(p => p.classList.remove("active"));

            item.classList.add("active");
            const activeView = document.getElementById(`page-${targetPage}`);
            if (activeView) {
                activeView.classList.add("active");
            }
        });
    });
}

// ---------------------------------------------------------
// 2. QUICK TEST SAMPLE BUTTONS
// ---------------------------------------------------------
function initSampleButtons() {
    const sampleBtns = document.querySelectorAll(".sample-btn");
    const reviewInput = document.getElementById("reviewInput");

    sampleBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const text = btn.getAttribute("data-text");
            if (reviewInput && text) {
                reviewInput.value = text;
                // Switch to analyze tab if not active
                document.querySelector('[data-page="analyze"]').click();
                analyzeReview();
            }
        });
    });
}

// ---------------------------------------------------------
// 3. API HEALTH CHECK
// ---------------------------------------------------------
async function checkApiHealth() {
    const healthBadge = document.getElementById("healthStatus");
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();

        if (response.ok && data.status === "ok") {
            if (healthBadge) {
                healthBadge.textContent = "🟢 Model Online";
                healthBadge.title = `Model: ${data.model} | Macro F1: ${data.macro_f1}`;
            }
        } else {
            if (healthBadge) {
                healthBadge.textContent = "🔴 Model Offline";
            }
        }
    } catch (err) {
        console.error("Health check error:", err);
        if (healthBadge) {
            healthBadge.textContent = "⚠️ Server Offline";
        }
    }
}

// ---------------------------------------------------------
// 4. ANALYZE REVIEW INFERENCE (HERO FEATURE)
// ---------------------------------------------------------
async function analyzeReview() {
    const inputEl = document.getElementById("reviewInput");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const resultContainer = document.getElementById("resultContainer");
    const errorMsg = document.getElementById("errorMsg");

    if (!inputEl) return;
    const reviewText = inputEl.value.trim();

    // Reset view states
    errorMsg.style.display = "none";
    errorMsg.textContent = "";

    if (!reviewText) {
        errorMsg.textContent = "⚠️ Please enter or paste a customer review before analyzing.";
        errorMsg.style.display = "block";
        return;
    }

    // Show loading state on button
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = `<span class="spinner"></span> Analyzing Review...`;

    try {
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ review: reviewText })
        });

        const resData = await response.json();

        if (!response.ok || resData.error) {
            throw new Error(resData.error || "Failed to process review.");
        }

        const data = resData.data;
        renderPredictionResult(data);
        resultContainer.style.display = "block";
        resultContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });

    } catch (err) {
        console.error("Inference Error:", err);
        errorMsg.textContent = `❌ Error: ${err.message}`;
        errorMsg.style.display = "block";
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = `🔍 Analyze Review`;
    }
}

// Render dynamic prediction output
function renderPredictionResult(data) {
    const badgeEl = document.getElementById("resSentimentBadge");
    const confScoreEl = document.getElementById("resConfScore");
    const confFillEl = document.getElementById("resConfFill");
    const origTextEl = document.getElementById("resOrigText");
    const cleanTextEl = document.getElementById("resCleanText");
    const aspectsGrid = document.getElementById("resAspectsGrid");
    const probList = document.getElementById("resProbList");

    const sentiment = data.sentiment || "Neutral";
    const confidence = Math.round((data.confidence || 0) * 100);

    // 1. Sentiment Badge & Styling
    badgeEl.className = "sentiment-badge-lg";
    if (sentiment === "Positive") {
        badgeEl.classList.add("positive");
        badgeEl.innerHTML = `✓ POSITIVE`;
    } else if (sentiment === "Negative") {
        badgeEl.classList.add("negative");
        badgeEl.innerHTML = `✕ NEGATIVE`;
    } else {
        badgeEl.classList.add("neutral");
        badgeEl.innerHTML = `! NEUTRAL`;
    }

    // 2. Confidence Bar
    confScoreEl.textContent = `${confidence}%`;
    confFillEl.style.width = `${confidence}%`;

    // 3. Review Texts
    origTextEl.textContent = data.original_text || "";
    cleanTextEl.textContent = data.clean_text || "";

    // 4. Probabilities Breakdown
    if (data.probabilities) {
        probList.innerHTML = Object.entries(data.probabilities)
            .map(([cls, prob]) => `
                <tr>
                    <td><strong>${cls}</strong></td>
                    <td style="text-align: right;">${(prob * 100).toFixed(1)}%</td>
                </tr>
            `).join("");
    }

    // 5. ABSA Aspects Grid
    if (data.aspects && data.aspects.length > 0) {
        aspectsGrid.innerHTML = data.aspects.map(asp => {
            const s = asp.sentiment;
            const cls = s === "Positive" ? "positive" : s === "Negative" ? "negative" : "neutral";
            const icon = s === "Positive" ? "✓" : s === "Negative" ? "✕" : "!";

            return `
                <div class="aspect-card ${cls}">
                    <div class="aspect-name">
                        <span>${asp.aspect}</span>
                        <span>${icon} ${s}</span>
                    </div>
                    <div class="aspect-clause">"${asp.matching_clause}"</div>
                </div>
            `;
        }).join("");
    } else {
        aspectsGrid.innerHTML = `<p style="color: var(--text-muted); font-style: italic;">No domain-specific aspect keywords detected in this review.</p>`;
    }
}

// ---------------------------------------------------------
// 5. LOAD & RENDER ANALYTICS DATA (CHARTS)
// ---------------------------------------------------------
let charts = {};

async function loadAnalyticsData() {
    try {
        const response = await fetch(`${API_BASE}/api/analytics`);
        const result = await response.json();

        if (!response.ok || !result.data) return;

        const data = result.data;

        // Render Overview KPIs
        if (data.kpis) {
            document.getElementById("kpiTotal").textContent = data.kpis.total_reviews.toLocaleString();
            document.getElementById("kpiPos").textContent = `${data.kpis.positive_pct}%`;
            document.getElementById("kpiNeu").textContent = `${data.kpis.neutral_pct}%`;
            document.getElementById("kpiNeg").textContent = `${data.kpis.negative_pct}%`;
        }

        // Render Charts
        renderOverviewCharts(data);
        renderAspectCharts(data);
        renderModelComparisonTable(data.model_comparison);

    } catch (err) {
        console.error("Failed to load analytics data:", err);
    }
}

function renderOverviewCharts(data) {
    // 1. Overview Sentiment Donut
    const ctxSent = document.getElementById("overviewSentimentChart");
    if (ctxSent) {
        if (charts.sentiment) charts.sentiment.destroy();
        charts.sentiment = new Chart(ctxSent, {
            type: 'doughnut',
            data: {
                labels: ['Positive (4-5★)', 'Negative (1-2★)', 'Neutral (3★)'],
                datasets: [{
                    data: [77.94, 14.50, 7.56],
                    backgroundColor: ['#10b981', '#ef4444', '#f59e0b']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom' } }
            }
        });
    }

    // 2. Rating Distribution Bar Chart
    const ctxRating = document.getElementById("overviewRatingChart");
    if (ctxRating && data.rating_distribution) {
        if (charts.rating) charts.rating.destroy();
        const labels = data.rating_distribution.map(r => r.rating);
        const counts = data.rating_distribution.map(r => r.count);

        charts.rating = new Chart(ctxRating, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Number of Reviews',
                    data: counts,
                    backgroundColor: '#2563eb',
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } }
            }
        });
    }
}

function renderAspectCharts(data) {
    if (!data.aspect_summary || data.aspect_summary.length === 0) return;

    const aspects = data.aspect_summary;
    const labels = aspects.map(a => a.Aspect);
    const posMentions = aspects.map(a => a["Positive Mentions"]);
    const negMentions = aspects.map(a => a["Negative Mentions"]);
    const neuMentions = aspects.map(a => a["Neutral Mentions"]);

    // 1. Top Mentioned Aspects
    const ctxAspects = document.getElementById("analyticsAspectChart");
    if (ctxAspects) {
        if (charts.aspects) charts.aspects.destroy();
        charts.aspects = new Chart(ctxAspects, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Total Mentions',
                    data: aspects.map(a => a["Total Mentions"]),
                    backgroundColor: '#6366f1',
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: { legend: { display: false } }
            }
        });
    }

    // 2. Aspect Sentiment Breakdown (Stacked)
    const ctxStack = document.getElementById("aspectComparisonChart");
    if (ctxStack) {
        if (charts.stack) charts.stack.destroy();
        charts.stack = new Chart(ctxStack, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    { label: 'Positive', data: posMentions, backgroundColor: '#10b981' },
                    { label: 'Neutral', data: neuMentions, backgroundColor: '#f59e0b' },
                    { label: 'Negative', data: negMentions, backgroundColor: '#ef4444' }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { x: { stacked: true }, y: { stacked: true } }
            }
        });
    }
}

function renderModelComparisonTable(models) {
    const tableBody = document.getElementById("modelTableBody");
    if (!tableBody || !models) return;

    tableBody.innerHTML = models.map(m => `
        <tr ${m.Model.includes("Logistic") ? 'style="font-weight: 700; background: #eff6ff;"' : ''}>
            <td>${m.Model} ${m.Model.includes("Logistic") ? '🏆 (Selected)' : ''}</td>
            <td>${m["Accuracy (%)"]}%</td>
            <td>${m["Macro F1-Score (%)"]}%</td>
            <td>${m["Negative F1 (%)"]}%</td>
            <td>${m["Positive F1 (%)"]}%</td>
        </tr>
    `).join("");
}
