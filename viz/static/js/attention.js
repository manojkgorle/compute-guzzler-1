/**
 * attention.js — Attention Inspector tab.
 * D3.js heatmaps for single-head and all-heads attention patterns.
 */

(function () {
    'use strict';

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    var attnState = {
        prompt: '',
        tokens: [],
        attentionData: null,    // full response from API
        selectedLayer: 0,
        selectedHead: 'all',
        entropyData: null,      // cached entropy from attention response
        entropySummary: null,   // cached entropy_summary from attention response
        ablationData: null,     // cached ablation response
        predictions: null,      // cached predictions from /api/predict
    };

    var layerSelect = null;
    var headSelect = null;

    // -----------------------------------------------------------------------
    // Init
    // -----------------------------------------------------------------------
    window.initAttention = function () {
        buildControls();
    };

    // -----------------------------------------------------------------------
    // Build controls
    // -----------------------------------------------------------------------
    function buildControls() {
        var container = document.getElementById('attention-controls');
        container.innerHTML = '';

        createPromptInput('attention-controls', handleAnalyze, {
            placeholder: 'The quick brown fox jumps over the lazy dog',
            buttonLabel: 'Analyze',
        });

        // Layer dropdown
        var layerOpts = [];
        for (var i = 0; i < 12; i++) {
            layerOpts.push({ value: String(i), label: 'Layer ' + i });
        }
        layerSelect = createDropdown('attention-controls', 'Layer:', 'attn-layer-select', layerOpts);
        layerSelect.addEventListener('change', function () {
            attnState.selectedLayer = parseInt(layerSelect.value);
            if (attnState.attentionData) renderAttention();
        });

        // Head dropdown (all + 0-11)
        var headOpts = [{ value: 'all', label: 'All heads' }];
        for (var j = 0; j < 12; j++) {
            headOpts.push({ value: String(j), label: 'Head ' + j });
        }
        headSelect = createDropdown('attention-controls', 'Head:', 'attn-head-select', headOpts);
        headSelect.addEventListener('change', function () {
            attnState.selectedHead = headSelect.value;
            if (attnState.attentionData) renderAttention();
        });
    }

    // -----------------------------------------------------------------------
    // API call
    // -----------------------------------------------------------------------
    async function handleAnalyze(prompt) {
        attnState.prompt = prompt;
        attnState.predictions = null;
        var viewContainer = document.getElementById('attention-view');
        viewContainer.innerHTML = '';

        showInlineLoading('attention-view');

        // Disable button while loading
        var btn = document.getElementById('attention-controls-prompt-btn');
        if (btn) btn.disabled = true;

        try {
            var params = { prompt: prompt };
            // Fetch attention data and predictions in parallel
            var [data, predData] = await Promise.all([
                apiFetch('/api/attention', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(params),
                }),
                apiFetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(params),
                }).catch(function () { return null; }),
            ]);

            attnState.attentionData = data;
            attnState.tokens = data.tokens || [];

            // Cache entropy data from the same response
            attnState.entropyData = data.entropy || null;
            attnState.entropySummary = data.entropy_summary || null;

            // Cache predictions
            if (predData && predData.predictions) {
                attnState.predictions = predData.predictions;
            }

            renderAttention();
        } catch (err) {
            showError('attention-view', 'Failed to fetch attention data: ' + err.message);
        } finally {
            if (btn) btn.disabled = false;
        }
    }

    // -----------------------------------------------------------------------
    // Render dispatcher
    // -----------------------------------------------------------------------
    function renderAttention() {
        var viewContainer = document.getElementById('attention-view');
        viewContainer.innerHTML = '';

        // Prediction display at the top
        if (attnState.predictions) {
            renderPredictionDisplay(viewContainer, attnState.predictions);
        }

        if (attnState.selectedHead === 'all') {
            renderAllHeads(viewContainer);
        } else {
            renderSingleHead(viewContainer, parseInt(attnState.selectedHead));
        }

        // Render entropy heatmap if data is available
        if (attnState.entropySummary) {
            renderEntropyHeatmap(viewContainer);
        }

        // Render ablation section (button + results if available)
        renderAblationSection(viewContainer);
    }

    // -----------------------------------------------------------------------
    // Prediction display: show model's top next-token prediction
    // -----------------------------------------------------------------------
    function renderPredictionDisplay(container, predictions) {
        if (!predictions || predictions.length === 0) return;

        var display = document.createElement('div');
        display.className = 'prediction-display';

        // Title
        var title = document.createElement('div');
        title.className = 'prediction-title';
        title.textContent = 'MODEL PREDICTION';
        display.appendChild(title);

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

        display.appendChild(mainRow);

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

            display.appendChild(runnersRow);
        }

        container.appendChild(display);
    }

    // -----------------------------------------------------------------------
    // Single head heatmap (D3)
    // -----------------------------------------------------------------------
    function renderSingleHead(container, headIdx) {
        var layer = attnState.selectedLayer;
        var weights = getAttentionWeights(layer, headIdx);
        var tokens = attnState.tokens;

        if (!weights || !tokens.length) {
            container.innerHTML = '<div class="placeholder-message">No attention data for this selection.</div>';
            return;
        }

        var n = tokens.length;
        var maxTotalSize = 620;
        var cellSize = Math.max(16, Math.min(48, Math.floor(maxTotalSize / n)));
        var labelSpace = 80;
        var width = n * cellSize + labelSpace + 40;
        var height = n * cellSize + labelSpace + 40;

        var wrapper = document.createElement('div');
        wrapper.className = 'heatmap-container';
        container.appendChild(wrapper);

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
            .attr('transform', 'translate(' + labelSpace + ',' + labelSpace + ')');

        var colorScale = d3.scaleSequential(d3.interpolateBlues).domain([0, 1]);

        // Cells
        for (var row = 0; row < n; row++) {
            for (var col = 0; col < n; col++) {
                var val = weights[row][col];
                var isMasked = col > row;
                var cellFill = isMasked ? '#1a1a2e' : colorScale(val);
                var cell = g.append('rect')
                    .attr('x', col * cellSize)
                    .attr('y', row * cellSize)
                    .attr('width', cellSize - 1)
                    .attr('height', cellSize - 1)
                    .attr('rx', 2)
                    .attr('fill', cellFill)
                    .attr('data-row', row)
                    .attr('data-col', col)
                    .attr('data-val', val.toFixed(4))
                    .attr('data-masked', isMasked ? '1' : '0');

                if (!isMasked) {
                    cell.style('cursor', 'pointer')
                        .on('mouseover', function (event) {
                            var r = +this.getAttribute('data-row');
                            var c = +this.getAttribute('data-col');
                            var v = this.getAttribute('data-val');
                            tooltip.style.display = 'block';
                            tooltip.innerHTML =
                                'Query: <b>' + escapeHtml(tokens[r]) + '</b> &rarr; Key: <b>' +
                                escapeHtml(tokens[c]) + '</b>, weight: ' + v;
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
        }

        // X-axis labels (key tokens, rotated)
        for (var ci = 0; ci < n; ci++) {
            g.append('text')
                .attr('x', ci * cellSize + cellSize / 2)
                .attr('y', -6)
                .attr('text-anchor', 'start')
                .attr('transform', 'rotate(-45, ' + (ci * cellSize + cellSize / 2) + ', -6)')
                .attr('fill', '#e0e0e0')
                .attr('font-size', Math.min(12, cellSize - 2) + 'px')
                .attr('font-family', 'Courier New, monospace')
                .text(truncateToken(tokens[ci], 8));
        }

        // Y-axis labels (query tokens)
        for (var ri = 0; ri < n; ri++) {
            g.append('text')
                .attr('x', -8)
                .attr('y', ri * cellSize + cellSize / 2 + 4)
                .attr('text-anchor', 'end')
                .attr('fill', '#e0e0e0')
                .attr('font-size', Math.min(12, cellSize - 2) + 'px')
                .attr('font-family', 'Courier New, monospace')
                .text(truncateToken(tokens[ri], 8));
        }

        // Title
        svg.append('text')
            .attr('x', width / 2)
            .attr('y', 20)
            .attr('text-anchor', 'middle')
            .attr('fill', '#e0e0e0')
            .attr('font-size', '14px')
            .attr('font-weight', '600')
            .text('Layer ' + layer + ', Head ' + headIdx);

        // Subtitle / explanation
        svg.append('text')
            .attr('x', width / 2)
            .attr('y', 36)
            .attr('text-anchor', 'middle')
            .attr('fill', '#888')
            .attr('font-size', '12px')
            .text('Each row shows where that token looks. Dark = high attention weight.');

        // Clean up tooltip on container removal
        var observer = new MutationObserver(function () {
            if (!document.body.contains(wrapper)) {
                tooltip.remove();
                observer.disconnect();
            }
        });
        observer.observe(container, { childList: true });
    }

    // -----------------------------------------------------------------------
    // All-heads mini heatmap grid
    // -----------------------------------------------------------------------
    function renderAllHeads(container) {
        var layer = attnState.selectedLayer;
        var tokens = attnState.tokens;

        var grid = document.createElement('div');
        grid.className = 'mini-heatmap-grid';
        container.appendChild(grid);

        for (var h = 0; h < 12; h++) {
            (function (headIdx) {
                var weights = getAttentionWeights(layer, headIdx);
                if (!weights) return;

                var card = document.createElement('div');
                card.className = 'mini-heatmap';
                card.title = 'Click to inspect Head ' + headIdx;

                var label = document.createElement('div');
                label.className = 'mini-label';
                label.textContent = 'Head ' + headIdx;
                card.appendChild(label);

                var n = tokens.length;
                var miniSize = Math.max(3, Math.min(8, Math.floor(140 / n)));
                var svgW = n * miniSize;
                var svgH = n * miniSize;

                var svgEl = d3.select(card)
                    .append('svg')
                    .attr('width', svgW)
                    .attr('height', svgH)
                    .style('display', 'block')
                    .style('margin', '0 auto');

                var colorScale = d3.scaleSequential(d3.interpolateBlues).domain([0, 1]);

                for (var row = 0; row < n; row++) {
                    for (var col = 0; col < n; col++) {
                        var isMasked = col > row;
                        svgEl.append('rect')
                            .attr('x', col * miniSize)
                            .attr('y', row * miniSize)
                            .attr('width', miniSize)
                            .attr('height', miniSize)
                            .attr('fill', isMasked ? '#1a1a2e' : colorScale(weights[row][col]));
                    }
                }

                // Click to switch to single-head view
                card.addEventListener('click', function () {
                    headSelect.value = String(headIdx);
                    attnState.selectedHead = String(headIdx);
                    renderAttention();
                });

                grid.appendChild(card);
            })(h);
        }
    }

    // -----------------------------------------------------------------------
    // Data extraction helpers
    // -----------------------------------------------------------------------
    function getAttentionWeights(layer, head) {
        var data = attnState.attentionData;
        if (!data || !data.layers) return null;

        // API structure: data.layers[layer][head] = 2D array (n x n)
        var layerData = data.layers[layer];
        if (!layerData) return null;

        if (Array.isArray(layerData[head])) {
            return layerData[head];
        }

        return null;
    }

    function truncateToken(token, maxLen) {
        if (!token) return '';
        if (token.length <= maxLen) return token;
        return token.slice(0, maxLen - 1) + '\u2026';
    }

    // -----------------------------------------------------------------------
    // Entropy heatmap (12x12 grid: layers x heads)
    // -----------------------------------------------------------------------
    function renderEntropyHeatmap(container) {
        var summary = attnState.entropySummary;
        if (!summary) return;

        var wrapper = document.createElement('div');
        wrapper.className = 'heatmap-container';
        wrapper.id = 'entropy-heatmap-container';
        container.appendChild(wrapper);

        var nLayers = 12;
        var nHeads = 12;
        var cellSize = 44;
        var labelSpaceLeft = 70;
        var labelSpaceTop = 50;
        var width = nHeads * cellSize + labelSpaceLeft + 20;
        var height = nLayers * cellSize + labelSpaceTop + 20;

        // Collect all values for color scale domain
        var allVals = [];
        for (var l = 0; l < nLayers; l++) {
            for (var h = 0; h < nHeads; h++) {
                var val = (summary[l] && summary[l][h] != null) ? summary[l][h] : 0;
                allVals.push(val);
            }
        }
        var minVal = d3.min(allVals);
        var maxVal = d3.max(allVals);

        var colorScale = d3.scaleSequential(d3.interpolateYlOrRd).domain([minVal, maxVal]);

        // Tooltip
        var tooltip = document.createElement('div');
        tooltip.className = 'heatmap-tooltip';
        tooltip.style.display = 'none';
        document.body.appendChild(tooltip);

        var svg = d3.select(wrapper)
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        // Title
        svg.append('text')
            .attr('x', width / 2)
            .attr('y', 18)
            .attr('text-anchor', 'middle')
            .attr('fill', '#e0e0e0')
            .attr('font-size', '14px')
            .attr('font-weight', '600')
            .text('Attention Head Entropy');

        var g = svg.append('g')
            .attr('transform', 'translate(' + labelSpaceLeft + ',' + labelSpaceTop + ')');

        // Cells
        for (var row = 0; row < nLayers; row++) {
            for (var col = 0; col < nHeads; col++) {
                var v = (summary[row] && summary[row][col] != null) ? summary[row][col] : 0;
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
                            '<br>Mean Entropy: <b>' + vl + '</b>';
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

    // -----------------------------------------------------------------------
    // Head Ablation section
    // -----------------------------------------------------------------------
    function renderAblationSection(container) {
        var section = document.createElement('div');
        section.id = 'ablation-section';
        section.style.marginTop = '24px';
        container.appendChild(section);

        // Title
        var title = document.createElement('div');
        title.className = 'chart-title';
        title.textContent = 'Head Ablation Importance (loss increase when head removed)';
        title.style.marginBottom = '12px';
        section.appendChild(title);

        // Run Ablation button
        var btn = document.createElement('button');
        btn.className = 'btn btn-primary';
        btn.id = 'ablation-run-btn';
        btn.textContent = 'Run Ablation';
        btn.style.marginBottom = '12px';
        section.appendChild(btn);

        // Results container
        var resultsDiv = document.createElement('div');
        resultsDiv.id = 'ablation-results';
        section.appendChild(resultsDiv);

        // If we already have ablation data for the same prompt, re-render it
        if (attnState.ablationData && attnState.ablationData._prompt === attnState.prompt) {
            renderAblationHeatmap(resultsDiv, attnState.ablationData);
        }

        btn.addEventListener('click', function () {
            handleRunAblation(btn, resultsDiv);
        });
    }

    async function handleRunAblation(btn, resultsDiv) {
        if (!attnState.prompt) return;

        btn.disabled = true;
        btn.textContent = 'Running ablation (144 forward passes)...';
        resultsDiv.innerHTML = '';
        showInlineLoading('ablation-results');

        try {
            var data = await apiFetch('/api/ablation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: attnState.prompt }),
            });

            data._prompt = attnState.prompt;
            attnState.ablationData = data;

            resultsDiv.innerHTML = '';
            renderAblationHeatmap(resultsDiv, data);
        } catch (err) {
            resultsDiv.innerHTML = '';
            showError('ablation-results', 'Ablation failed: ' + err.message);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Run Ablation';
        }
    }

    function renderAblationHeatmap(container, data) {
        var importance = data.importance; // 12x12 array
        if (!importance) return;

        // Summary info
        var infoDiv = document.createElement('div');
        infoDiv.className = 'summary-text';
        infoDiv.style.marginBottom = '12px';
        var infoHtml = 'Baseline loss: <b>' + (data.baseline_loss != null ? data.baseline_loss.toFixed(4) : '?') + '</b>';
        if (data.max_importance) {
            infoHtml += ' &mdash; Most important: <b>Layer ' + data.max_importance.layer +
                ', Head ' + data.max_importance.head + '</b> (delta: +' +
                data.max_importance.delta.toFixed(4) + ')';
        }
        infoDiv.innerHTML = infoHtml;
        container.appendChild(infoDiv);

        var wrapper = document.createElement('div');
        wrapper.className = 'heatmap-container';
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
                allVals.push(importance[l][h]);
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
                var v = importance[row][col];
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
                            '<br>Loss delta: <b>' + vl + '</b>';
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
})();
