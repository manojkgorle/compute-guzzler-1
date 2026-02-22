/**
 * attribution.js — Logit Attribution tab.
 * Token-level logit attribution, per-layer contribution breakdown,
 * and logit lens table.
 */

(function () {
    'use strict';

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    var attrState = {
        prompt: '',
        data: null,           // full API response
        tokens: [],
        selectedPos: null,    // clicked token position index
        dlaData: null,        // cached DLA response
    };

    // -----------------------------------------------------------------------
    // Init
    // -----------------------------------------------------------------------
    window.initAttribution = function () {
        buildControls();
    };

    // -----------------------------------------------------------------------
    // Build controls
    // -----------------------------------------------------------------------
    function buildControls() {
        var container = document.getElementById('attribution-controls');
        container.innerHTML = '';

        createPromptInput('attribution-controls', handleAnalyze, {
            placeholder: 'The Eiffel Tower is located in',
            buttonLabel: 'Analyze',
        });
    }

    // -----------------------------------------------------------------------
    // API call
    // -----------------------------------------------------------------------
    async function handleAnalyze(prompt) {
        attrState.prompt = prompt;
        attrState.selectedPos = null;

        // Clear previous prediction display
        var existingPrediction = document.getElementById('attribution-prediction');
        if (existingPrediction) existingPrediction.remove();

        var tokenStrip = document.getElementById('attribution-token-strip');
        tokenStrip.innerHTML = '';
        var detail = document.getElementById('attribution-detail');
        detail.innerHTML = '';

        showInlineLoading('attribution-token-strip');

        var btn = document.getElementById('attribution-controls-prompt-btn');
        if (btn) btn.disabled = true;

        try {
            var data = await apiFetch('/api/attribution', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: prompt, top_k: 10 }),
            });

            attrState.data = data;
            attrState.tokens = data.tokens || [];
            renderPredictionDisplay(data.next_token_predictions || []);
            renderTokenStrip();
        } catch (err) {
            showError('attribution-token-strip', 'Failed to fetch attribution data: ' + err.message);
        } finally {
            if (btn) btn.disabled = false;
        }
    }

    // -----------------------------------------------------------------------
    // Prediction display: show model's top next-token prediction
    // -----------------------------------------------------------------------
    function renderPredictionDisplay(predictions) {
        // Remove any existing prediction display
        var existing = document.getElementById('attribution-prediction');
        if (existing) existing.remove();

        if (!predictions || predictions.length === 0) return;

        var container = document.createElement('div');
        container.id = 'attribution-prediction';
        container.className = 'prediction-display';

        // Title
        var title = document.createElement('div');
        title.className = 'prediction-title';
        title.textContent = 'MODEL PREDICTION';
        container.appendChild(title);

        // Main prediction (top-1)
        var top = predictions[0];
        var mainRow = document.createElement('div');
        mainRow.className = 'prediction-main';

        var label = document.createElement('span');
        label.className = 'prediction-label';
        label.textContent = 'Next token:';
        mainRow.appendChild(label);

        var tokenBox = document.createElement('span');
        tokenBox.className = 'prediction-token-main';
        tokenBox.textContent = top.token;
        mainRow.appendChild(tokenBox);

        var probBadge = document.createElement('span');
        probBadge.className = 'prediction-prob-main';
        probBadge.textContent = (top.prob * 100).toFixed(1) + '%';
        mainRow.appendChild(probBadge);

        container.appendChild(mainRow);

        // Runner-up predictions (2-5)
        if (predictions.length > 1) {
            var runnersRow = document.createElement('div');
            runnersRow.className = 'prediction-runners';

            var runnersLabel = document.createElement('span');
            runnersLabel.className = 'prediction-label';
            runnersLabel.textContent = 'Also likely:';
            runnersRow.appendChild(runnersLabel);

            var limit = Math.min(predictions.length, 5);
            for (var i = 1; i < limit; i++) {
                var pred = predictions[i];
                var pill = document.createElement('span');
                pill.className = 'prediction-pill';
                pill.innerHTML =
                    '<span class="prediction-pill-token">' + escapeHtml(pred.token) + '</span>' +
                    '<span class="prediction-pill-prob">' + (pred.prob * 100).toFixed(1) + '%</span>';
                runnersRow.appendChild(pill);
            }

            container.appendChild(runnersRow);
        }

        // Insert above the token strip
        var tokenStrip = document.getElementById('attribution-token-strip');
        tokenStrip.parentNode.insertBefore(container, tokenStrip);
    }

    // -----------------------------------------------------------------------
    // Token strip: colored by prediction confidence
    // -----------------------------------------------------------------------
    function renderTokenStrip() {
        var container = document.getElementById('attribution-token-strip');
        container.innerHTML = '';

        var tokens = attrState.tokens;
        var positions = attrState.data.positions || [];

        tokens.forEach(function (token, idx) {
            var span = document.createElement('span');
            span.className = 'token-span';
            span.textContent = token;
            span.dataset.pos = idx;

            // Color: high confidence = green, low = red
            var prob = (positions[idx] && positions[idx].final_prob != null) ? positions[idx].final_prob : 0.5;
            var bg = probabilityColor(prob);
            span.style.backgroundColor = bg;
            span.style.color = prob > 0.5 ? '#1a1a2e' : '#e0e0e0';

            span.addEventListener('click', function () {
                selectPosition(idx);
            });

            container.appendChild(span);
        });

        // Auto-select last predictable position if available
        if (tokens.length > 1) {
            selectPosition(tokens.length - 1);
        }
    }

    // -----------------------------------------------------------------------
    // Position selection
    // -----------------------------------------------------------------------
    function selectPosition(pos) {
        attrState.selectedPos = pos;

        // Update visual selection
        document.querySelectorAll('#attribution-token-strip .token-span').forEach(function (el) {
            el.classList.toggle('selected', parseInt(el.dataset.pos) === pos);
        });

        renderDetail(pos);
    }

    // -----------------------------------------------------------------------
    // Render detail view for a selected position
    // -----------------------------------------------------------------------
    function renderDetail(pos) {
        var detail = document.getElementById('attribution-detail');
        detail.innerHTML = '';
        detail.className = 'chart-grid';

        var data = attrState.data;
        if (!data) return;

        // Get attribution data for this position
        var posData = null;
        if (data.positions && data.positions[pos]) {
            posData = data.positions[pos];
        }

        if (!posData) {
            detail.innerHTML = '<div class="placeholder-message">No attribution data for this position.</div>';
            return;
        }

        // 1. Stacked bar chart: per-layer logit contributions
        var barCard = document.createElement('div');
        barCard.className = 'chart-card full-width';
        barCard.innerHTML =
            '<div class="chart-title">Per-Layer Logit Contribution</div>' +
            '<div class="chart-container" id="attr-bar-chart"></div>';
        detail.appendChild(barCard);

        // 2. Logit lens table
        var tableCard = document.createElement('div');
        tableCard.className = 'chart-card full-width';
        tableCard.innerHTML =
            '<div class="chart-title">Logit Lens: Top Predictions at Each Depth</div>' +
            '<div id="attr-lens-table" style="overflow-x:auto;"></div>';
        detail.appendChild(tableCard);

        // 3. Per-Head DLA section
        var dlaCard = document.createElement('div');
        dlaCard.className = 'chart-card full-width';
        dlaCard.innerHTML =
            '<div class="chart-title">Direct Logit Attribution (per-head)</div>' +
            '<div id="attr-dla-section"></div>';
        detail.appendChild(dlaCard);

        // Render charts
        renderContributionBar(posData);
        renderLogitLensTable(posData);
        renderSummary(detail, posData);
        renderDLASection(pos);
    }

    // -----------------------------------------------------------------------
    // 1. Per-layer logit contribution bar chart
    // -----------------------------------------------------------------------
    function renderContributionBar(posData) {
        var contributionsObj = posData.layer_contributions || {};
        var targetToken = posData.target || '?';

        // API returns layer_contributions as dict: {"embedding": float, "0": float, ...}
        // Convert to ordered array: [embedding, layer0, layer1, ..., layer11]
        var labels = ['Embed'];
        var values = [contributionsObj['embedding'] || 0];
        for (var i = 0; i < 12; i++) {
            labels.push('L' + i);
            values.push(contributionsObj[String(i)] || 0);
        }

        var colors = values.map(function (v) {
            return v >= 0 ? '#4ecca3' : '#e94560';
        });

        var traces = [
            {
                x: labels,
                y: values,
                type: 'bar',
                marker: { color: colors },
                hovertemplate: '%{x}<br>Contribution: %{y:+.3f} logits<extra></extra>',
            },
        ];

        var layout = darkLayout({
            xaxis: { title: 'Layer', gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
            yaxis: { title: 'Logit Contribution', gridcolor: '#2a2a4a', zerolinecolor: '#e94560' },
            showlegend: false,
            title: {
                text: 'Contributions to predicting "' + targetToken + '"',
                font: { size: 13, color: '#8892a4' },
                x: 0.5,
            },
            shapes: [
                {
                    type: 'line',
                    x0: -0.5,
                    x1: 12.5,
                    y0: 0,
                    y1: 0,
                    line: { color: '#8892a4', width: 1, dash: 'dash' },
                },
            ],
        });

        Plotly.newPlot('attr-bar-chart', traces, layout, window.PLOTLY_CONFIG);
    }

    // -----------------------------------------------------------------------
    // 2. Logit lens table
    // -----------------------------------------------------------------------
    function renderLogitLensTable(posData) {
        var container = document.getElementById('attr-lens-table');
        if (!container) return;

        // API returns cumulative_predictions as dict: {"embedding": [{token, prob}...], "0": [...], ...}
        // Convert to array of {depth, top_tokens} for table rendering
        var cumulPred = posData.cumulative_predictions || {};
        var targetToken = posData.target || '';

        var lensData = [];
        if (cumulPred['embedding']) {
            lensData.push({ depth: 'Embed', top_tokens: cumulPred['embedding'] });
        }
        for (var li = 0; li < 12; li++) {
            if (cumulPred[String(li)]) {
                lensData.push({ depth: 'Layer ' + li, top_tokens: cumulPred[String(li)] });
            }
        }

        if (!lensData.length) {
            container.innerHTML = '<div class="placeholder-message">No logit lens data available.</div>';
            return;
        }

        var table = document.createElement('table');
        table.className = 'logit-lens-table';

        // Header
        var maxCols = 5;
        var thead = document.createElement('thead');
        var headerRow = document.createElement('tr');
        var depthTh = document.createElement('th');
        depthTh.textContent = 'Depth';
        headerRow.appendChild(depthTh);

        for (var c = 0; c < maxCols; c++) {
            var th = document.createElement('th');
            th.textContent = '#' + (c + 1) + ' Prediction';
            headerRow.appendChild(th);
        }
        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Body
        var tbody = document.createElement('tbody');

        lensData.forEach(function (row) {
            var tr = document.createElement('tr');

            var depthTd = document.createElement('td');
            depthTd.textContent = row.depth || '';
            depthTd.style.fontWeight = '600';
            depthTd.style.textAlign = 'left';
            tr.appendChild(depthTd);

            var topTokens = row.top_tokens || [];
            for (var j = 0; j < maxCols; j++) {
                var td = document.createElement('td');
                if (topTokens[j]) {
                    var tok = topTokens[j].token || '';
                    var prob = topTokens[j].prob;
                    var probStr = prob != null ? (prob * 100).toFixed(1) + '%' : '';

                    td.innerHTML =
                        '<span style="font-weight:600;">' + escapeHtml(tok) + '</span>' +
                        '<br><span style="font-size:11px;color:#8892a4;">' + probStr + '</span>';

                    // Highlight if this is the target token
                    if (tok === targetToken) {
                        td.className = 'highlight-token';
                    }
                }
                tr.appendChild(td);
            }

            tbody.appendChild(tr);
        });

        table.appendChild(tbody);
        container.innerHTML = '';
        container.appendChild(table);
    }

    // -----------------------------------------------------------------------
    // 3. Summary text
    // -----------------------------------------------------------------------
    function renderSummary(detailContainer, posData) {
        var contributionsObj = posData.layer_contributions || {};
        var targetToken = posData.target || '?';

        // Convert dict to ordered array for finding max
        var labels = ['Embedding'];
        var contributions = [contributionsObj['embedding'] || 0];
        for (var i = 0; i < 12; i++) {
            labels.push('Layer ' + i);
            contributions.push(contributionsObj[String(i)] || 0);
        }

        if (Object.keys(contributionsObj).length === 0) return;

        // Find layer with max positive contribution
        var maxVal = -Infinity;
        var maxIdx = 0;

        contributions.forEach(function (v, idx) {
            if (v > maxVal) {
                maxVal = v;
                maxIdx = idx;
            }
        });

        var summary = document.createElement('div');
        summary.className = 'summary-text';
        summary.innerHTML =
            labels[maxIdx] + ' contributes the most (<b>+' + maxVal.toFixed(2) +
            ' logits</b>) to predicting "<b>' + escapeHtml(targetToken) + '</b>"';

        detailContainer.appendChild(summary);
    }

    // -----------------------------------------------------------------------
    // Color helpers
    // -----------------------------------------------------------------------
    function probabilityColor(prob) {
        // Interpolate from red (low) through yellow (mid) to green (high)
        var r, g, b;
        if (prob < 0.5) {
            // Red to yellow
            var t = prob / 0.5;
            r = Math.round(233 - t * 33);   // 233 -> 200
            g = Math.round(69 + t * 131);   // 69  -> 200
            b = Math.round(96 - t * 56);    // 96  -> 40
        } else {
            // Yellow to green
            var t2 = (prob - 0.5) / 0.5;
            r = Math.round(200 - t2 * 122);  // 200 -> 78
            g = Math.round(200 + t2 * 4);    // 200 -> 204
            b = Math.round(40 + t2 * 123);   // 40  -> 163
        }
        return 'rgba(' + r + ',' + g + ',' + b + ',0.7)';
    }

    // -----------------------------------------------------------------------
    // Per-Head DLA section
    // -----------------------------------------------------------------------
    function renderDLASection(pos) {
        var section = document.getElementById('attr-dla-section');
        if (!section) return;

        section.innerHTML = '';

        // Button to run DLA
        var btn = document.createElement('button');
        btn.className = 'btn btn-primary';
        btn.id = 'dla-run-btn';
        btn.textContent = 'Per-Head DLA';
        btn.style.marginBottom = '12px';
        section.appendChild(btn);

        // Results container
        var resultsDiv = document.createElement('div');
        resultsDiv.id = 'dla-results';
        section.appendChild(resultsDiv);

        // If we already have DLA data for same prompt+position, show it
        if (attrState.dlaData &&
            attrState.dlaData._prompt === attrState.prompt &&
            attrState.dlaData._position === pos) {
            renderDLAResults(resultsDiv, attrState.dlaData);
        }

        btn.addEventListener('click', function () {
            handleRunDLA(btn, resultsDiv, pos);
        });
    }

    async function handleRunDLA(btn, resultsDiv, pos) {
        if (!attrState.prompt) return;

        btn.disabled = true;
        btn.textContent = 'Computing per-head DLA...';
        resultsDiv.innerHTML = '';
        showInlineLoading('dla-results');

        try {
            var data = await apiFetch('/api/dla', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: attrState.prompt, position: pos }),
            });

            data._prompt = attrState.prompt;
            data._position = pos;
            attrState.dlaData = data;

            resultsDiv.innerHTML = '';
            renderDLAResults(resultsDiv, data);
        } catch (err) {
            resultsDiv.innerHTML = '';
            showError('dla-results', 'DLA failed: ' + err.message);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Per-Head DLA';
        }
    }

    function renderDLAResults(container, data) {
        // Summary info
        var summaryDiv = document.createElement('div');
        summaryDiv.className = 'summary-text';
        summaryDiv.style.marginBottom = '12px';
        var summaryHtml = 'Target: "<b>' + escapeHtml(data.target || '?') + '</b>"' +
            ' (prob: ' + (data.final_prob != null ? (data.final_prob * 100).toFixed(1) + '%' : '?') + ')' +
            ' &mdash; Embedding contribution: <b>' +
            (data.embedding_contribution != null ? data.embedding_contribution.toFixed(3) : '?') +
            '</b> &mdash; Total reconstructed: <b>' +
            (data.total_reconstructed != null ? data.total_reconstructed.toFixed(3) : '?') + '</b>';
        summaryDiv.innerHTML = summaryHtml;
        container.appendChild(summaryDiv);

        // Head contributions heatmap (D3)
        if (data.head_contributions) {
            renderDLAHeadHeatmap(container, data.head_contributions);
        }

        // MLP contributions bar chart (Plotly)
        if (data.mlp_contributions) {
            renderDLAMlpBar(container, data.mlp_contributions);
        }
    }

    function renderDLAHeadHeatmap(container, headContributions) {
        var wrapper = document.createElement('div');
        wrapper.className = 'heatmap-container';
        wrapper.style.marginBottom = '16px';
        container.appendChild(wrapper);

        var nLayers = 12;
        var nHeads = 12;
        var cellSize = 44;
        var labelSpaceLeft = 70;
        var labelSpaceTop = 30;
        var width = nHeads * cellSize + labelSpaceLeft + 20;
        var height = nLayers * cellSize + labelSpaceTop + 20;

        // Find extent for diverging scale
        var allVals = [];
        for (var l = 0; l < nLayers; l++) {
            for (var h = 0; h < nHeads; h++) {
                allVals.push(headContributions[l][h]);
            }
        }
        var absMax = d3.max(allVals.map(function (v) { return Math.abs(v); }));
        if (absMax === 0) absMax = 1;

        // Diverging scale: blue (negative) -> white (zero) -> red (positive)
        var colorScale = d3.scaleDiverging(d3.interpolateRdBu)
            .domain([absMax, 0, -absMax]);

        // Tooltip
        var tooltip = document.createElement('div');
        tooltip.className = 'heatmap-tooltip';
        tooltip.style.display = 'none';
        document.body.appendChild(tooltip);

        var svg = d3.select(wrapper)
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        var g = svg.append('g')
            .attr('transform', 'translate(' + labelSpaceLeft + ',' + labelSpaceTop + ')');

        // Cells
        for (var row = 0; row < nLayers; row++) {
            for (var col = 0; col < nHeads; col++) {
                var v = headContributions[row][col];
                g.append('rect')
                    .attr('x', col * cellSize)
                    .attr('y', row * cellSize)
                    .attr('width', cellSize - 2)
                    .attr('height', cellSize - 2)
                    .attr('rx', 3)
                    .attr('fill', colorScale(v))
                    .attr('data-layer', row)
                    .attr('data-head', col)
                    .attr('data-val', v.toFixed(4))
                    .style('cursor', 'pointer')
                    .on('mouseover', function () {
                        var lr = this.getAttribute('data-layer');
                        var hd = this.getAttribute('data-head');
                        var vl = this.getAttribute('data-val');
                        tooltip.style.display = 'block';
                        tooltip.innerHTML =
                            'Layer ' + lr + ', Head ' + hd +
                            '<br>Contribution: <b>' + vl + '</b> logits';
                    })
                    .on('mousemove', function (event) {
                        tooltip.style.left = (event.clientX + 12) + 'px';
                        tooltip.style.top = (event.clientY - 10) + 'px';
                    })
                    .on('mouseout', function () {
                        tooltip.style.display = 'none';
                    });
            }
        }

        // X-axis labels (heads)
        for (var ci = 0; ci < nHeads; ci++) {
            g.append('text')
                .attr('x', ci * cellSize + cellSize / 2)
                .attr('y', -8)
                .attr('text-anchor', 'middle')
                .attr('fill', '#e0e0e0')
                .attr('font-size', '11px')
                .text('H' + ci);
        }

        // Y-axis labels (layers)
        for (var ri = 0; ri < nLayers; ri++) {
            g.append('text')
                .attr('x', -8)
                .attr('y', ri * cellSize + cellSize / 2 + 4)
                .attr('text-anchor', 'end')
                .attr('fill', '#e0e0e0')
                .attr('font-size', '11px')
                .text('L' + ri);
        }

        // Clean up tooltip on container removal
        var observer = new MutationObserver(function () {
            if (!document.body.contains(wrapper)) {
                tooltip.remove();
                observer.disconnect();
            }
        });
        observer.observe(container, { childList: true });
    }

    function renderDLAMlpBar(container, mlpContributions) {
        var chartDiv = document.createElement('div');
        chartDiv.id = 'attr-dla-mlp-chart';
        chartDiv.style.width = '100%';
        chartDiv.style.height = '300px';
        container.appendChild(chartDiv);

        var labels = [];
        var values = [];
        for (var i = 0; i < mlpContributions.length; i++) {
            labels.push('L' + i);
            values.push(mlpContributions[i]);
        }

        var colors = values.map(function (v) {
            return v >= 0 ? '#4ecca3' : '#e94560';
        });

        var traces = [
            {
                x: labels,
                y: values,
                type: 'bar',
                marker: { color: colors },
                hovertemplate: '%{x}<br>MLP Contribution: %{y:+.3f} logits<extra></extra>',
            },
        ];

        var layout = darkLayout({
            xaxis: { title: 'Layer', gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
            yaxis: { title: 'MLP Contribution (logits)', gridcolor: '#2a2a4a', zerolinecolor: '#e94560' },
            showlegend: false,
            title: {
                text: 'MLP Contributions per Layer',
                font: { size: 13, color: '#8892a4' },
                x: 0.5,
            },
            shapes: [
                {
                    type: 'line',
                    x0: -0.5,
                    x1: 11.5,
                    y0: 0,
                    y1: 0,
                    line: { color: '#8892a4', width: 1, dash: 'dash' },
                },
            ],
        });

        Plotly.newPlot(chartDiv, traces, layout, window.PLOTLY_CONFIG);
    }
})();
