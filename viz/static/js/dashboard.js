/**
 * dashboard.js — Training Dashboard tab.
 * Shows live metrics: loss curves, LR schedule, gradient norms,
 * per-layer residual norms, and MLP sparsity.
 */

(function () {
    'use strict';

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    var state = {
        steps: [],
        trainLoss: [],
        valSteps: [],
        valLoss: [],
        perplexity: [],
        lr: [],
        gradNorm: [],
        layerResNorms: null,   // { 0: [], 1: [], ... }
        layerMlpSparsity: null,
        layerGradNorms: null,  // { 0: [], 1: [], ... } per-layer gradient norms
        latestLoss: null,
        latestPerplexity: null,
        latestLR: null,
    };

    var LAYER_COLORS = [
        '#e94560', '#f0a500', '#4ecca3', '#3282b8',
        '#bb86fc', '#03dac6', '#ff7597', '#ffd166',
        '#06d6a0', '#118ab2', '#ef476f', '#8338ec',
    ];

    // -----------------------------------------------------------------------
    // Init
    // -----------------------------------------------------------------------
    window.initDashboard = function () {
        buildLayout();
        fetchAllMetrics();
    };

    // -----------------------------------------------------------------------
    // Build DOM layout
    // -----------------------------------------------------------------------
    function buildLayout() {
        // Stat cards
        var statsContainer = document.getElementById('dashboard-stats');
        statsContainer.innerHTML = [
            statCardHtml('dash-loss', 'Current Loss', '--'),
            statCardHtml('dash-ppl', 'Current Perplexity', '--'),
            statCardHtml('dash-lr', 'Current LR', '--'),
        ].join('');

        // Charts
        var chartsContainer = document.getElementById('dashboard-charts');
        chartsContainer.innerHTML = [
            chartCardHtml('dash-loss-chart', 'Loss Curve', true),
            '<div class="chart-grid two-col" id="dash-row2">',
            chartCardHtml('dash-lr-chart', 'Learning Rate Schedule'),
            chartCardHtml('dash-grad-chart', 'Gradient Norm'),
            '</div>',
            '<div class="chart-grid two-col" id="dash-row3">',
            chartCardHtml('dash-resnorm-chart', 'Per-Layer Residual Norms'),
            chartCardHtml('dash-sparsity-chart', 'Per-Layer MLP Sparsity'),
            '</div>',
            '<div class="chart-grid two-col" id="dash-row4">',
            chartCardHtml('dash-layer-grad-chart', 'Per-Layer Gradient Norms'),
            '</div>',
        ].join('');
    }

    function statCardHtml(id, label, value) {
        return (
            '<div class="stat-card">' +
            '<div class="stat-label">' + label + '</div>' +
            '<div class="stat-value" id="' + id + '">' + value + '</div>' +
            '</div>'
        );
    }

    function chartCardHtml(id, title, fullWidth) {
        return (
            '<div class="chart-card' + (fullWidth ? ' full-width' : '') + '">' +
            '<div class="chart-title">' + title + '</div>' +
            '<div class="chart-container" id="' + id + '"></div>' +
            '</div>'
        );
    }

    // -----------------------------------------------------------------------
    // Fetch all metrics on init
    // -----------------------------------------------------------------------
    async function fetchAllMetrics() {
        showLoading('dashboard-charts');
        try {
            var data = await apiFetch('/api/metrics/all');
            populateState(data);
            renderAllCharts();
        } catch (err) {
            console.warn('[dashboard] fetch error:', err.message);
            showPlaceholder('dash-loss-chart', 'No training data yet. Start a training run to see metrics.');
        } finally {
            hideLoading('dashboard-charts');
        }
    }

    function populateState(data) {
        state.steps = data.steps || [];
        state.trainLoss = data.train_loss || [];
        state.valSteps = data.val_steps || [];
        state.valLoss = data.val_loss || [];
        state.perplexity = data.perplexity || [];
        state.lr = data.lr || [];
        state.gradNorm = data.grad_norm || [];
        state.layerResNorms = data.layer_res_norms || null;
        state.layerMlpSparsity = data.layer_mlp_sparsity || null;
        state.layerGradNorms = data.layer_grad_norms || null;

        if (state.trainLoss.length > 0) {
            state.latestLoss = state.trainLoss[state.trainLoss.length - 1];
        }
        if (state.perplexity.length > 0) {
            state.latestPerplexity = state.perplexity[state.perplexity.length - 1];
        }
        if (state.lr.length > 0) {
            state.latestLR = state.lr[state.lr.length - 1];
        }

        updateStatCards();
    }

    // -----------------------------------------------------------------------
    // Stat card updates
    // -----------------------------------------------------------------------
    function updateStatCards() {
        setStatValue('dash-loss', state.latestLoss, 4);
        setStatValue('dash-ppl', state.latestPerplexity, 1);
        setStatValue('dash-lr', state.latestLR, 6);
    }

    function setStatValue(id, value, decimals) {
        var el = document.getElementById(id);
        if (!el) return;
        if (value == null) {
            el.textContent = '--';
        } else {
            el.textContent = Number(value).toFixed(decimals);
        }
    }

    // -----------------------------------------------------------------------
    // Render all charts
    // -----------------------------------------------------------------------
    function renderAllCharts() {
        renderLossChart();
        renderLRChart();
        renderGradNormChart();
        renderResNormChart();
        renderSparsityChart();
        renderLayerGradNormChart();
    }

    // ---- Loss curve (train + val + perplexity on y2) ----
    function renderLossChart() {
        var traces = [
            {
                x: state.steps,
                y: state.trainLoss,
                name: 'Train Loss',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#3282b8', width: 2 },
                yaxis: 'y',
            },
            {
                x: state.valSteps,
                y: state.valLoss,
                name: 'Val Loss',
                type: 'scatter',
                mode: 'markers',
                marker: { color: '#e94560', size: 7, symbol: 'circle' },
                yaxis: 'y',
            },
            {
                x: state.steps,
                y: state.perplexity,
                name: 'Perplexity',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#4ecca3', width: 1.5, dash: 'dot' },
                yaxis: 'y2',
            },
        ];

        var layout = darkLayout({
            xaxis: { title: 'Step', gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
            yaxis: { title: 'Loss', gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
            yaxis2: {
                title: 'Perplexity',
                overlaying: 'y',
                side: 'right',
                gridcolor: 'rgba(42,42,74,0.3)',
                showgrid: false,
            },
            legend: {
                x: 0.01, y: 0.99,
                bgcolor: 'rgba(22,33,62,0.8)',
                bordercolor: '#2a2a4a',
                borderwidth: 1,
                font: { size: 11 },
            },
        });

        Plotly.newPlot('dash-loss-chart', traces, layout, window.PLOTLY_CONFIG);
    }

    // ---- LR Schedule ----
    function renderLRChart() {
        var traces = [
            {
                x: state.steps,
                y: state.lr,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#bb86fc', width: 2 },
                name: 'Learning Rate',
            },
        ];

        var layout = darkLayout({
            xaxis: { title: 'Step', gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
            yaxis: { title: 'LR', gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a', exponentformat: 'e' },
            showlegend: false,
        });

        Plotly.newPlot('dash-lr-chart', traces, layout, window.PLOTLY_CONFIG);
    }

    // ---- Gradient Norm ----
    function renderGradNormChart() {
        var traces = [
            {
                x: state.steps,
                y: state.gradNorm,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#f0a500', width: 2 },
                name: 'Grad Norm',
            },
        ];

        var layout = darkLayout({
            xaxis: { title: 'Step', gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
            yaxis: { title: 'Gradient L2 Norm', gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
            showlegend: false,
            shapes: [
                {
                    type: 'line',
                    x0: 0,
                    x1: 1,
                    xref: 'paper',
                    y0: 1.0,
                    y1: 1.0,
                    line: { color: '#e94560', width: 1.5, dash: 'dash' },
                },
            ],
        });

        Plotly.newPlot('dash-grad-chart', traces, layout, window.PLOTLY_CONFIG);
    }

    // ---- Per-layer residual norms (12 lines) ----
    function renderResNormChart() {
        if (!state.layerResNorms) {
            showPlaceholder('dash-resnorm-chart', 'No per-layer data available.');
            return;
        }

        var traces = [];
        for (var i = 0; i < 12; i++) {
            var key = String(i);
            if (!state.layerResNorms[key]) continue;
            traces.push({
                x: state.steps,
                y: state.layerResNorms[key],
                type: 'scatter',
                mode: 'lines',
                line: { color: LAYER_COLORS[i], width: 1.5 },
                name: 'Layer ' + i,
            });
        }

        var layout = darkLayout({
            xaxis: { title: 'Step', gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
            yaxis: { title: 'Residual L2 Norm', gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
            legend: {
                x: 1.02, y: 1,
                bgcolor: 'rgba(22,33,62,0.8)',
                bordercolor: '#2a2a4a',
                borderwidth: 1,
                font: { size: 10 },
            },
        });

        Plotly.newPlot('dash-resnorm-chart', traces, layout, window.PLOTLY_CONFIG);
    }

    // ---- Per-layer MLP sparsity (12 lines) ----
    function renderSparsityChart() {
        if (!state.layerMlpSparsity) {
            showPlaceholder('dash-sparsity-chart', 'No per-layer data available.');
            return;
        }

        var traces = [];
        for (var i = 0; i < 12; i++) {
            var key = String(i);
            if (!state.layerMlpSparsity[key]) continue;
            traces.push({
                x: state.steps,
                y: state.layerMlpSparsity[key],
                type: 'scatter',
                mode: 'lines',
                line: { color: LAYER_COLORS[i], width: 1.5 },
                name: 'Layer ' + i,
            });
        }

        var layout = darkLayout({
            xaxis: { title: 'Step', gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
            yaxis: {
                title: 'MLP Sparsity (% zeros)',
                gridcolor: '#2a2a4a',
                zerolinecolor: '#2a2a4a',
                range: [0, 1],
                tickformat: '.0%',
            },
            legend: {
                x: 1.02, y: 1,
                bgcolor: 'rgba(22,33,62,0.8)',
                bordercolor: '#2a2a4a',
                borderwidth: 1,
                font: { size: 10 },
            },
        });

        Plotly.newPlot('dash-sparsity-chart', traces, layout, window.PLOTLY_CONFIG);
    }

    // ---- Per-layer gradient norms (12 lines) ----
    function renderLayerGradNormChart() {
        if (!state.layerGradNorms) {
            showPlaceholder('dash-layer-grad-chart', 'No per-layer gradient norm data available.');
            return;
        }

        var traces = [];
        for (var i = 0; i < 12; i++) {
            var key = String(i);
            if (!state.layerGradNorms[key]) continue;
            traces.push({
                x: state.steps,
                y: state.layerGradNorms[key],
                type: 'scatter',
                mode: 'lines',
                line: { color: LAYER_COLORS[i], width: 1.5 },
                name: 'Layer ' + i,
            });
        }

        var layout = darkLayout({
            xaxis: { title: 'Step', gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
            yaxis: { title: 'Gradient L2 Norm', gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
            legend: {
                x: 1.02, y: 1,
                bgcolor: 'rgba(22,33,62,0.8)',
                bordercolor: '#2a2a4a',
                borderwidth: 1,
                font: { size: 10 },
            },
        });

        Plotly.newPlot('dash-layer-grad-chart', traces, layout, window.PLOTLY_CONFIG);
    }

    // -----------------------------------------------------------------------
    // WebSocket handlers
    // -----------------------------------------------------------------------
    window.handleStepUpdate = function (data) {
        // data: { step, train_loss, perplexity, lr, grad_norm, layer_res_norms?, layer_mlp_sparsity? }
        state.steps.push(data.step);
        state.trainLoss.push(data.train_loss);
        state.perplexity.push(data.perplexity);
        state.lr.push(data.lr);
        state.gradNorm.push(data.grad_norm);

        state.latestLoss = data.train_loss;
        state.latestPerplexity = data.perplexity;
        state.latestLR = data.lr;

        // Per-layer data
        if (data.layer_res_norms) {
            if (!state.layerResNorms) state.layerResNorms = {};
            for (var k in data.layer_res_norms) {
                if (!state.layerResNorms[k]) state.layerResNorms[k] = [];
                state.layerResNorms[k].push(data.layer_res_norms[k]);
            }
        }
        if (data.layer_mlp_sparsity) {
            if (!state.layerMlpSparsity) state.layerMlpSparsity = {};
            for (var k in data.layer_mlp_sparsity) {
                if (!state.layerMlpSparsity[k]) state.layerMlpSparsity[k] = [];
                state.layerMlpSparsity[k].push(data.layer_mlp_sparsity[k]);
            }
        }
        if (data.per_layer_grad_norms) {
            if (!state.layerGradNorms) state.layerGradNorms = {};
            for (var i = 0; i < data.per_layer_grad_norms.length; i++) {
                var key = String(i);
                if (!state.layerGradNorms[key]) state.layerGradNorms[key] = [];
                state.layerGradNorms[key].push(data.per_layer_grad_norms[i]);
            }
        }

        updateStatCards();

        // Only extend plots if dashboard tab is active (avoids invisible Plotly ops)
        if (window.vizState.activeTab !== 'dashboard') return;

        // Extend loss chart
        var lossDiv = document.getElementById('dash-loss-chart');
        if (lossDiv && lossDiv.data) {
            Plotly.extendTraces('dash-loss-chart', {
                x: [[data.step], [], [data.step]],
                y: [[data.train_loss], [], [data.perplexity]],
            }, [0, 1, 2]);
        }

        // Extend LR chart
        var lrDiv = document.getElementById('dash-lr-chart');
        if (lrDiv && lrDiv.data) {
            Plotly.extendTraces('dash-lr-chart', {
                x: [[data.step]],
                y: [[data.lr]],
            }, [0]);
        }

        // Extend grad norm chart
        var gradDiv = document.getElementById('dash-grad-chart');
        if (gradDiv && gradDiv.data) {
            Plotly.extendTraces('dash-grad-chart', {
                x: [[data.step]],
                y: [[data.grad_norm]],
            }, [0]);
        }

        // Re-render per-layer charts (extend is tricky with dynamic traces)
        if (data.layer_res_norms) renderResNormChart();
        if (data.layer_mlp_sparsity) renderSparsityChart();
        if (data.per_layer_grad_norms) renderLayerGradNormChart();
    };

    window.handleValUpdate = function (data) {
        // data: { step, val_loss }
        state.valSteps.push(data.step);
        state.valLoss.push(data.val_loss);

        if (window.vizState.activeTab !== 'dashboard') return;

        var lossDiv = document.getElementById('dash-loss-chart');
        if (lossDiv && lossDiv.data) {
            Plotly.extendTraces('dash-loss-chart', {
                x: [[], [data.step], []],
                y: [[], [data.val_loss], []],
            }, [0, 1, 2]);
        }
    };

    // -----------------------------------------------------------------------
    // Placeholder message
    // -----------------------------------------------------------------------
    function showPlaceholder(containerId, message) {
        var el = document.getElementById(containerId);
        if (!el) return;
        el.innerHTML = '<div class="placeholder-message">' + escapeHtml(message) + '</div>';
    }
})();
