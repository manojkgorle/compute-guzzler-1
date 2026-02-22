/**
 * activations.js — Activation Explorer tab.
 * Residual norms, MLP activation distributions, top neurons,
 * and attention-vs-MLP output norm comparison.
 */

(function () {
    'use strict';

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    var actState = {
        prompt: '',
        data: null,       // full API response
        selectedLayer: 0,
    };

    var layerSelect = null;

    // -----------------------------------------------------------------------
    // Init
    // -----------------------------------------------------------------------
    window.initActivations = function () {
        buildControls();
        buildChartSlots();
    };

    // -----------------------------------------------------------------------
    // Build controls
    // -----------------------------------------------------------------------
    function buildControls() {
        var container = document.getElementById('activations-controls');
        container.innerHTML = '';

        createPromptInput('activations-controls', handleAnalyze, {
            placeholder: 'The capital of France is Paris',
            buttonLabel: 'Analyze',
        });

        var layerOpts = [];
        for (var i = 0; i < 12; i++) {
            layerOpts.push({ value: String(i), label: 'Layer ' + i });
        }
        layerSelect = createDropdown('activations-controls', 'Layer:', 'act-layer-select', layerOpts);
        layerSelect.addEventListener('change', function () {
            actState.selectedLayer = parseInt(layerSelect.value);
            if (actState.data) renderLayerCharts();
        });
    }

    // -----------------------------------------------------------------------
    // Build chart card slots
    // -----------------------------------------------------------------------
    function buildChartSlots() {
        var chartsContainer = document.getElementById('activations-charts');
        chartsContainer.innerHTML = '';
        chartsContainer.className = 'chart-grid two-col';

        var cards = [
            { id: 'act-resnorm', title: 'Residual Stream Norm per Position' },
            { id: 'act-mlp-hist', title: 'MLP Hidden Activation Histogram' },
            { id: 'act-top-neurons', title: 'Top-20 Most Active Neurons' },
            { id: 'act-attn-vs-mlp', title: 'Attention vs MLP Output Norm (All Layers)' },
        ];

        cards.forEach(function (card) {
            var div = document.createElement('div');
            div.className = 'chart-card';
            div.innerHTML =
                '<div class="chart-title">' + card.title + '</div>' +
                '<div class="chart-container" id="' + card.id + '"></div>';
            chartsContainer.appendChild(div);
        });
    }

    // -----------------------------------------------------------------------
    // API call
    // -----------------------------------------------------------------------
    async function handleAnalyze(prompt) {
        actState.prompt = prompt;

        var btn = document.getElementById('activations-controls-prompt-btn');
        if (btn) btn.disabled = true;

        showLoading('activations-charts');

        try {
            var data = await apiFetch('/api/activations', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: prompt }),
            });

            actState.data = data;
            renderAllCharts();
        } catch (err) {
            showError('act-resnorm', 'Failed to fetch activations: ' + err.message);
        } finally {
            hideLoading('activations-charts');
            if (btn) btn.disabled = false;
        }
    }

    // -----------------------------------------------------------------------
    // Render all charts
    // -----------------------------------------------------------------------
    function renderAllCharts() {
        renderLayerCharts();
        renderAttnVsMlp();
    }

    // Charts that depend on the selected layer
    function renderLayerCharts() {
        renderResNorm();
        renderMlpHistogram();
        renderTopNeurons();
    }

    // -----------------------------------------------------------------------
    // 1. Residual stream norm per position (bar chart)
    // -----------------------------------------------------------------------
    function renderResNorm() {
        var data = actState.data;
        if (!data) return;

        var layer = actState.selectedLayer;
        var tokens = data.tokens || [];
        var norms = null;

        // API structure: data.layers[layer].residual_norms = array of floats per position
        if (data.layers && data.layers[layer] && data.layers[layer].residual_norms) {
            norms = data.layers[layer].residual_norms;
        }

        if (!norms) {
            showPlaceholder('act-resnorm', 'No residual norm data available.');
            return;
        }

        var traces = [
            {
                x: tokens,
                y: norms,
                type: 'bar',
                marker: {
                    color: norms.map(function (v) {
                        return 'rgba(50, 130, 184, ' + Math.min(1, 0.3 + v / (Math.max.apply(null, norms) || 1) * 0.7) + ')';
                    }),
                },
                hovertemplate: 'Token: %{x}<br>L2 Norm: %{y:.3f}<extra></extra>',
            },
        ];

        var layout = darkLayout({
            xaxis: {
                title: 'Token',
                gridcolor: '#2a2a4a',
                zerolinecolor: '#2a2a4a',
                tickangle: -45,
                tickfont: { family: 'Courier New, monospace', size: 11 },
            },
            yaxis: { title: 'L2 Norm', gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
            showlegend: false,
            title: { text: 'Layer ' + layer, font: { size: 13, color: '#8892a4' }, x: 0.5 },
        });

        Plotly.newPlot('act-resnorm', traces, layout, window.PLOTLY_CONFIG);
    }

    // -----------------------------------------------------------------------
    // 2. MLP hidden activation histogram
    // -----------------------------------------------------------------------
    function renderMlpHistogram() {
        var data = actState.data;
        if (!data) return;

        var layer = actState.selectedLayer;
        var histogram = null;

        // API structure: data.layers[layer].mlp_hidden_stats.histogram = { bins: [...], counts: [...] }
        if (data.layers && data.layers[layer] &&
            data.layers[layer].mlp_hidden_stats &&
            data.layers[layer].mlp_hidden_stats.histogram) {
            histogram = data.layers[layer].mlp_hidden_stats.histogram;
        }

        if (!histogram || !histogram.bins || !histogram.counts) {
            showPlaceholder('act-mlp-hist', 'No MLP activation data available.');
            return;
        }

        var traces = [
            {
                x: histogram.bins,
                y: histogram.counts,
                type: 'bar',
                marker: { color: '#4ecca3', line: { color: '#1a1a2e', width: 0.5 } },
                opacity: 0.85,
                hovertemplate: 'Value: %{x:.3f}<br>Count: %{y}<extra></extra>',
            },
        ];

        var layout = darkLayout({
            xaxis: { title: 'Post-GELU Activation Value', gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
            yaxis: { title: 'Count', gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
            showlegend: false,
            bargap: 0.02,
            title: { text: 'Layer ' + layer, font: { size: 13, color: '#8892a4' }, x: 0.5 },
        });

        Plotly.newPlot('act-mlp-hist', traces, layout, window.PLOTLY_CONFIG);
    }

    // -----------------------------------------------------------------------
    // 3. Top-20 most active neurons (horizontal bar chart)
    // -----------------------------------------------------------------------
    function renderTopNeurons() {
        var data = actState.data;
        if (!data) return;

        var layer = actState.selectedLayer;
        var topNeurons = null;

        // API structure: data.layers[layer].mlp_hidden_stats.top_neurons = [{idx, mean_activation}, ...]
        if (data.layers && data.layers[layer] &&
            data.layers[layer].mlp_hidden_stats &&
            data.layers[layer].mlp_hidden_stats.top_neurons) {
            topNeurons = data.layers[layer].mlp_hidden_stats.top_neurons;
        }

        if (!topNeurons || !topNeurons.length) {
            showPlaceholder('act-top-neurons', 'No top neuron data available.');
            return;
        }

        var labels = topNeurons.map(function (n) { return 'N' + n.idx; });
        var values = topNeurons.map(function (n) { return n.mean_activation; });

        // Reverse so largest is at top
        var revLabels = labels.slice().reverse();
        var revValues = values.slice().reverse();

        var traces = [
            {
                y: revLabels,
                x: revValues,
                type: 'bar',
                orientation: 'h',
                marker: {
                    color: revValues.map(function (v) {
                        return v >= 0 ? '#4ecca3' : '#e94560';
                    }),
                },
                hovertemplate: 'Neuron: %{y}<br>Activation: %{x:.4f}<extra></extra>',
            },
        ];

        var layout = darkLayout({
            xaxis: { title: 'Mean Activation', gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
            yaxis: {
                title: '',
                gridcolor: '#2a2a4a',
                zerolinecolor: '#2a2a4a',
                tickfont: { family: 'Courier New, monospace', size: 10 },
            },
            showlegend: false,
            margin: { l: 60, r: 20, t: 36, b: 48 },
            title: { text: 'Layer ' + layer, font: { size: 13, color: '#8892a4' }, x: 0.5 },
        });

        Plotly.newPlot('act-top-neurons', traces, layout, window.PLOTLY_CONFIG);
    }

    // -----------------------------------------------------------------------
    // 4. Attention vs MLP output norm comparison (grouped bar, all layers)
    // -----------------------------------------------------------------------
    function renderAttnVsMlp() {
        var data = actState.data;
        if (!data) return;

        // API structure: data.layers[layer].attn_output_norms and .mlp_output_norms are per-position arrays
        // We compute the mean norm across positions for each layer
        if (!data.layers) {
            showPlaceholder('act-attn-vs-mlp', 'No layer comparison data available.');
            return;
        }

        var layerLabels = [];
        var attnNorms = [];
        var mlpNorms = [];

        for (var i = 0; i < 12; i++) {
            layerLabels.push('L' + i);
            var layerData = data.layers[i];
            if (layerData && layerData.attn_output_norms && layerData.mlp_output_norms) {
                var attnArr = layerData.attn_output_norms;
                var mlpArr = layerData.mlp_output_norms;
                var attnMean = attnArr.reduce(function (a, b) { return a + b; }, 0) / (attnArr.length || 1);
                var mlpMean = mlpArr.reduce(function (a, b) { return a + b; }, 0) / (mlpArr.length || 1);
                attnNorms.push(attnMean);
                mlpNorms.push(mlpMean);
            } else {
                attnNorms.push(0);
                mlpNorms.push(0);
            }
        }

        if (attnNorms.every(function (v) { return v === 0; })) {
            showPlaceholder('act-attn-vs-mlp', 'No layer comparison data available.');
            return;
        }

        var traces = [
            {
                x: layerLabels,
                y: attnNorms,
                name: 'Attention',
                type: 'bar',
                marker: { color: '#3282b8' },
                hovertemplate: 'Layer %{x}<br>Attn Norm: %{y:.3f}<extra></extra>',
            },
            {
                x: layerLabels,
                y: mlpNorms,
                name: 'MLP',
                type: 'bar',
                marker: { color: '#e94560' },
                hovertemplate: 'Layer %{x}<br>MLP Norm: %{y:.3f}<extra></extra>',
            },
        ];

        var layout = darkLayout({
            xaxis: { title: 'Layer', gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
            yaxis: { title: 'Output L2 Norm', gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
            barmode: 'group',
            legend: {
                x: 0.01, y: 0.99,
                bgcolor: 'rgba(22,33,62,0.8)',
                bordercolor: '#2a2a4a',
                borderwidth: 1,
                font: { size: 11 },
            },
        });

        Plotly.newPlot('act-attn-vs-mlp', traces, layout, window.PLOTLY_CONFIG);
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------
    function showPlaceholder(containerId, message) {
        var el = document.getElementById(containerId);
        if (!el) return;
        el.innerHTML = '<div class="placeholder-message">' + escapeHtml(message) + '</div>';
    }
})();
