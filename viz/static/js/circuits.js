/**
 * circuits.js — Circuits tab: causal intervention experiments.
 *
 * Four subsections inspired by "On the Biology of a Large Language Model":
 *   1. Activation Patching — patch clean activations into corrupted run
 *   2. Activation Steering — scale components and observe output shift
 *   3. Activation Swapping — transplant activations between prompts
 *   4. Pre-computation Detection — find where future tokens appear early
 */

(function () {
    'use strict';

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    var circState = {
        steeringPrompt: '',
        steeringDebounce: null,
    };

    // -----------------------------------------------------------------------
    // Init
    // -----------------------------------------------------------------------
    window.initCircuits = function () {
        var container = document.getElementById('circuits-container');
        container.innerHTML = '';

        buildPatchingSection(container);
        buildSteeringSection(container);
        buildSwappingSection(container);
        buildPrecomputationSection(container);

        // Expand the first section by default
        var first = container.querySelector('.circuits-subsection');
        if (first) first.classList.add('expanded');
    };

    // -----------------------------------------------------------------------
    // Helper: create a collapsible subsection
    // -----------------------------------------------------------------------
    function createSubsection(parent, id, title, tag, description) {
        var section = document.createElement('div');
        section.className = 'circuits-subsection';
        section.id = id;

        var header = document.createElement('div');
        header.className = 'circuits-subsection-header';
        header.innerHTML =
            '<div style="display:flex;align-items:center;gap:10px;">' +
            '<h3>' + escapeHtml(title) + '</h3>' +
            '<span class="subsection-tag">' + escapeHtml(tag) + '</span>' +
            '</div>' +
            '<span class="expand-icon">&#9660;</span>';

        header.addEventListener('click', function () {
            section.classList.toggle('expanded');
        });

        var body = document.createElement('div');
        body.className = 'circuits-subsection-body';

        if (description) {
            var desc = document.createElement('div');
            desc.className = 'circuits-subsection-desc';
            desc.innerHTML = description;
            body.appendChild(desc);
        }

        section.appendChild(header);
        section.appendChild(body);
        parent.appendChild(section);

        return body;
    }

    // -----------------------------------------------------------------------
    // Helper: render prediction list in a comparison panel
    // -----------------------------------------------------------------------
    function renderPredictionPanel(container, title, predictions, color) {
        var panel = document.createElement('div');
        panel.className = 'comparison-panel';

        var titleEl = document.createElement('div');
        titleEl.className = 'panel-title';
        titleEl.textContent = title;
        panel.appendChild(titleEl);

        if (!predictions || predictions.length === 0) {
            panel.innerHTML += '<div class="placeholder-message">No predictions</div>';
            container.appendChild(panel);
            return;
        }

        var maxProb = predictions[0].prob || 0.01;

        predictions.forEach(function (pred) {
            var row = document.createElement('div');
            row.className = 'pred-row';

            var token = document.createElement('span');
            token.className = 'pred-token';
            token.textContent = pred.token;
            row.appendChild(token);

            var barContainer = document.createElement('div');
            barContainer.className = 'pred-bar-container';
            var bar = document.createElement('div');
            bar.className = 'pred-bar';
            bar.style.width = Math.max(2, (pred.prob / maxProb) * 100) + '%';
            bar.style.background = color || 'var(--success)';
            barContainer.appendChild(bar);
            row.appendChild(barContainer);

            var prob = document.createElement('span');
            prob.className = 'pred-prob';
            prob.textContent = (pred.prob * 100).toFixed(1) + '%';
            row.appendChild(prob);

            panel.appendChild(row);
        });

        container.appendChild(panel);
    }

    // -----------------------------------------------------------------------
    // Helper: create KL badge
    // -----------------------------------------------------------------------
    function createKLBadge(klValue) {
        var badge = document.createElement('div');
        badge.className = 'kl-badge';
        badge.innerHTML =
            '<span class="kl-label">KL Divergence:</span>' +
            '<span class="kl-value">' + klValue.toFixed(4) + '</span>';
        return badge;
    }

    // =======================================================================
    // 1. ACTIVATION PATCHING
    // =======================================================================
    function buildPatchingSection(parent) {
        var body = createSubsection(
            parent,
            'circuits-patching',
            'Activation Patching',
            'Causal Tracing',
            'Patch clean activations into a corrupted forward pass to find which ' +
            'layers and components are causally responsible for the correct prediction. ' +
            'Inspired by <em>Locating and Editing Factual Associations</em> (Meng et al., 2022).' +
            '<br><b>Tip:</b> Prompts must have the same token count for proper alignment. ' +
            'Change only a single-token word (e.g., "Japan" &rarr; "China", not "Haiti" which splits into 3 subwords).'
        );

        // Dual prompt inputs
        var dualGroup = document.createElement('div');
        dualGroup.className = 'dual-prompt-group';

        var cleanCol = document.createElement('div');
        cleanCol.className = 'prompt-col';
        cleanCol.innerHTML =
            '<label>Clean Prompt</label>' +
            '<input type="text" id="patching-clean-input" placeholder="In 1945 the war ended in">';

        var corruptedCol = document.createElement('div');
        corruptedCol.className = 'prompt-col';
        corruptedCol.innerHTML =
            '<label>Corrupted Prompt</label>' +
            '<input type="text" id="patching-corrupted-input" placeholder="In 1945 the war began in">';

        dualGroup.appendChild(cleanCol);
        dualGroup.appendChild(corruptedCol);
        body.appendChild(dualGroup);

        var btn = document.createElement('button');
        btn.className = 'btn btn-primary';
        btn.id = 'patching-run-btn';
        btn.textContent = 'Run Patching';
        body.appendChild(btn);

        var results = document.createElement('div');
        results.className = 'circuits-results';
        results.id = 'patching-results';
        body.appendChild(results);

        btn.addEventListener('click', function () {
            handlePatching(btn, results);
        });

        // Enter key on inputs
        ['patching-clean-input', 'patching-corrupted-input'].forEach(function (id) {
            document.getElementById(id).addEventListener('keydown', function (e) {
                if (e.key === 'Enter') handlePatching(btn, results);
            });
        });
    }

    async function handlePatching(btn, resultsDiv) {
        var cleanPrompt = document.getElementById('patching-clean-input').value.trim();
        var corruptedPrompt = document.getElementById('patching-corrupted-input').value.trim();
        if (!cleanPrompt || !corruptedPrompt) return;

        btn.disabled = true;
        btn.textContent = 'Running patching (38 forward passes)...';
        resultsDiv.innerHTML = '';
        showInlineLoading('patching-results');

        try {
            var data = await apiFetch('/api/circuits/patching', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    clean_prompt: cleanPrompt,
                    corrupted_prompt: corruptedPrompt,
                }),
            });

            resultsDiv.innerHTML = '';
            renderPatchingResults(resultsDiv, data);
        } catch (err) {
            resultsDiv.innerHTML = '';
            showError('patching-results', 'Patching failed: ' + err.message);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Run Patching';
        }
    }

    function renderPatchingResults(container, data) {
        // Token count mismatch warning
        if (data.token_count_mismatch) {
            var warning = document.createElement('div');
            warning.className = 'error-message';
            warning.style.marginBottom = '16px';
            warning.innerHTML =
                '<b>Token count mismatch:</b> Clean prompt has ' +
                data.clean_token_count + ' tokens, corrupted has ' +
                data.corrupted_token_count + '. ' +
                'Activation patching requires identical token counts for positional alignment. ' +
                'Results may be unreliable. Try prompts where only a single-token word differs ' +
                '(e.g., "Japan" / "China" / "Paris" / "India" are all single tokens, but "Haiti" splits into 3).';
            container.appendChild(warning);
        }

        // Summary info
        var logitGap = data.logit_gap || ((data.clean_logit || 0) - (data.corrupted_logit || 0));
        var summary = document.createElement('div');
        summary.className = 'summary-text';
        summary.style.marginBottom = '16px';
        summary.innerHTML =
            'Clean (' + data.clean_token_count + ' tokens) predicts "<b>' +
            escapeHtml(data.clean_pred || '?') + '</b>" ' +
            '(logit: ' + (data.clean_logit != null ? data.clean_logit.toFixed(2) : '?') + ') &mdash; ' +
            'Corrupted (' + data.corrupted_token_count + ' tokens) logit for same token: ' +
            (data.corrupted_logit != null ? data.corrupted_logit.toFixed(2) : '?') +
            ' &mdash; Gap: <b>' + logitGap.toFixed(2) + '</b>' +
            '<br>Max recovery: <b>Layer ' + data.max_recovery.layer +
            ' ' + data.max_recovery.component + '</b> (' +
            (data.max_recovery.recovery * 100).toFixed(1) + '% recovered)';
        container.appendChild(summary);

        // Heatmap: 12 layers x 3 components
        var results = data.patching_results || [];
        var components = ['residual', 'attn', 'mlp'];
        var compLabels = ['Residual', 'Attention', 'MLP'];
        var layerLabels = [];
        var zData = [[], [], []]; // 3 rows (components) x 12 cols (layers)
        var textData = [[], [], []]; // cell annotations

        // Collect all values to find range
        var allVals = [];
        results.forEach(function (r) {
            layerLabels.push('L' + r.layer);
            for (var ci = 0; ci < components.length; ci++) {
                var val = r[components[ci]] || 0;
                zData[ci].push(val);
                textData[ci].push((val * 100).toFixed(0) + '%');
                allVals.push(val);
            }
        });

        // Dynamic symmetric range, clamped to [-2, 2] to avoid extreme outliers dominating
        var absMax = Math.min(2, Math.max.apply(null, allVals.map(function (v) { return Math.abs(v); })));
        if (absMax < 0.1) absMax = 0.1;

        var chartDiv = document.createElement('div');
        chartDiv.id = 'patching-heatmap';
        chartDiv.style.width = '100%';
        chartDiv.style.height = '280px';
        container.appendChild(chartDiv);

        var traces = [{
            z: zData,
            x: layerLabels,
            y: compLabels,
            text: textData,
            texttemplate: '%{text}',
            textfont: { color: '#e0e0e0', size: 11 },
            type: 'heatmap',
            colorscale: [
                [0, '#2166ac'],
                [0.25, '#67a9cf'],
                [0.5, '#1a1a2e'],
                [0.75, '#f0a500'],
                [1, '#4ecca3'],
            ],
            zmin: -absMax,
            zmax: absMax,
            hovertemplate: '%{y} at %{x}<br>Recovery: %{z:.1%}<extra></extra>',
            colorbar: {
                title: { text: 'Recovery', font: { color: '#e0e0e0', size: 12 } },
                tickformat: '.0%',
                tickfont: { color: '#e0e0e0' },
            },
        }];

        var layout = darkLayout({
            title: { text: 'Logit Recovery by Layer and Component', font: { size: 14, color: '#8892a4' }, x: 0.5 },
            xaxis: { title: 'Layer', gridcolor: '#2a2a4a' },
            yaxis: { gridcolor: '#2a2a4a' },
            margin: { t: 40, r: 100, b: 48, l: 80 },
        });

        Plotly.newPlot(chartDiv, traces, layout, window.PLOTLY_CONFIG);

        // Raw logit delta bar chart (grouped by component)
        var deltaChartDiv = document.createElement('div');
        deltaChartDiv.id = 'patching-delta-chart';
        deltaChartDiv.style.width = '100%';
        deltaChartDiv.style.height = '280px';
        deltaChartDiv.style.marginTop = '16px';
        container.appendChild(deltaChartDiv);

        var deltaTraces = [];
        var compColors = { 'Residual': '#67a9cf', 'Attention': '#e94560', 'MLP': '#4ecca3' };
        for (var ci = 0; ci < components.length; ci++) {
            var deltas = [];
            results.forEach(function (r) {
                deltas.push(r[components[ci] + '_logit_delta'] || 0);
            });
            deltaTraces.push({
                x: layerLabels,
                y: deltas,
                name: compLabels[ci],
                type: 'bar',
                marker: { color: compColors[compLabels[ci]] },
                hovertemplate: compLabels[ci] + ' at %{x}<br>Logit delta: %{y:+.3f}<extra></extra>',
            });
        }

        var deltaLayout = darkLayout({
            title: { text: 'Raw Logit Change (patched \u2212 corrupted)', font: { size: 14, color: '#8892a4' }, x: 0.5 },
            xaxis: { title: 'Layer', gridcolor: '#2a2a4a' },
            yaxis: { title: 'Logit Delta', gridcolor: '#2a2a4a', zerolinecolor: '#8892a4' },
            barmode: 'group',
            legend: { bgcolor: 'rgba(22,33,62,0.8)', bordercolor: '#2a2a4a', borderwidth: 1, font: { size: 11 } },
            margin: { t: 40, r: 24, b: 48, l: 56 },
        });

        Plotly.newPlot(deltaChartDiv, deltaTraces, deltaLayout, window.PLOTLY_CONFIG);

        // Interpretation note when gap is small
        if (Math.abs(logitGap) < 2.0) {
            var note = document.createElement('div');
            note.className = 'circuits-subsection-desc';
            note.style.marginTop = '12px';
            note.innerHTML = '<b>Note:</b> The logit gap between clean and corrupted is small (' +
                logitGap.toFixed(2) + '). This means the model predicts similarly for both prompts, ' +
                'making recovery percentages volatile. The bar chart below shows raw logit deltas ' +
                'which are more stable. Try prompts where the model has a stronger ' +
                'distinction (e.g., different sentence structures, not just entity swaps).';
            container.appendChild(note);
        }
    }

    // =======================================================================
    // 2. ACTIVATION STEERING
    // =======================================================================
    function buildSteeringSection(parent) {
        var body = createSubsection(
            parent,
            'circuits-steering',
            'Activation Steering',
            'Ablation & Amplification',
            'Scale any attention head or MLP layer output and observe how the model\'s ' +
            'predictions change in real time. Scale=0 ablates, 1=unchanged, >1 amplifies.'
        );

        // Prompt input
        var promptGroup = document.createElement('div');
        promptGroup.className = 'prompt-group';
        promptGroup.style.marginBottom = '12px';
        promptGroup.innerHTML =
            '<input type="text" class="prompt-input" id="steering-prompt-input" ' +
            'placeholder="The cat sat on the">' +
            '<button class="btn btn-primary" id="steering-run-btn">Steer</button>';
        body.appendChild(promptGroup);

        // Controls row
        var controls = document.createElement('div');
        controls.className = 'circuits-controls-row';

        // Layer dropdown
        var layerGroup = document.createElement('div');
        layerGroup.className = 'select-group';
        layerGroup.innerHTML = '<label>Layer:</label>';
        var layerSelect = document.createElement('select');
        layerSelect.className = 'select-input';
        layerSelect.id = 'steering-layer';
        for (var i = 0; i < 12; i++) {
            var opt = document.createElement('option');
            opt.value = i;
            opt.textContent = 'Layer ' + i;
            layerSelect.appendChild(opt);
        }
        layerGroup.appendChild(layerSelect);
        controls.appendChild(layerGroup);

        // Component radio
        var radioGroup = document.createElement('div');
        radioGroup.className = 'radio-group';
        radioGroup.innerHTML =
            '<label><input type="radio" name="steering-comp" value="head" checked> Head</label>' +
            '<label><input type="radio" name="steering-comp" value="mlp"> MLP</label>';
        controls.appendChild(radioGroup);

        // Head dropdown
        var headGroup = document.createElement('div');
        headGroup.className = 'select-group';
        headGroup.id = 'steering-head-group';
        headGroup.innerHTML = '<label>Head:</label>';
        var headSelect = document.createElement('select');
        headSelect.className = 'select-input';
        headSelect.id = 'steering-head';
        for (var h = 0; h < 12; h++) {
            var hopt = document.createElement('option');
            hopt.value = h;
            hopt.textContent = 'Head ' + h;
            headSelect.appendChild(hopt);
        }
        headGroup.appendChild(headSelect);
        controls.appendChild(headGroup);

        // Scale slider
        var sliderGroup = document.createElement('div');
        sliderGroup.className = 'slider-group';
        sliderGroup.innerHTML =
            '<label>Scale:</label>' +
            '<input type="range" id="steering-scale" min="0" max="3" step="0.1" value="0">' +
            '<span class="slider-value" id="steering-scale-value">0.0</span>';
        controls.appendChild(sliderGroup);

        body.appendChild(controls);

        // Results
        var results = document.createElement('div');
        results.className = 'circuits-results';
        results.id = 'steering-results';
        body.appendChild(results);

        // Event handlers
        var runBtn = document.getElementById('steering-run-btn');
        runBtn.addEventListener('click', function () {
            runSteering(results);
        });

        document.getElementById('steering-prompt-input').addEventListener('keydown', function (e) {
            if (e.key === 'Enter') runSteering(results);
        });

        // Toggle head dropdown visibility
        radioGroup.addEventListener('change', function () {
            var comp = document.querySelector('input[name="steering-comp"]:checked').value;
            document.getElementById('steering-head-group').style.display =
                comp === 'head' ? 'flex' : 'none';
        });

        // Scale slider live update
        var scaleSlider = document.getElementById('steering-scale');
        scaleSlider.addEventListener('input', function () {
            document.getElementById('steering-scale-value').textContent =
                parseFloat(scaleSlider.value).toFixed(1);

            // Debounced re-run if we have a prompt
            if (circState.steeringPrompt) {
                clearTimeout(circState.steeringDebounce);
                circState.steeringDebounce = setTimeout(function () {
                    runSteering(results);
                }, 300);
            }
        });
    }

    async function runSteering(resultsDiv) {
        var prompt = document.getElementById('steering-prompt-input').value.trim();
        if (!prompt) return;
        circState.steeringPrompt = prompt;

        var layer = parseInt(document.getElementById('steering-layer').value);
        var component = document.querySelector('input[name="steering-comp"]:checked').value;
        var head = parseInt(document.getElementById('steering-head').value);
        var scale = parseFloat(document.getElementById('steering-scale').value);

        var btn = document.getElementById('steering-run-btn');
        btn.disabled = true;

        resultsDiv.innerHTML = '';
        showInlineLoading('steering-results');

        try {
            var data = await apiFetch('/api/circuits/steering', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: prompt,
                    layer: layer,
                    component: component,
                    head: head,
                    scale: scale,
                }),
            });

            resultsDiv.innerHTML = '';
            renderSteeringResults(resultsDiv, data);
        } catch (err) {
            resultsDiv.innerHTML = '';
            showError('steering-results', 'Steering failed: ' + err.message);
        } finally {
            btn.disabled = false;
        }
    }

    function renderSteeringResults(container, data) {
        // KL badge
        container.appendChild(createKLBadge(data.kl_divergence || 0));

        // Label
        var label = document.createElement('div');
        label.className = 'circuits-subsection-desc';
        var target = data.component === 'head'
            ? 'L' + data.layer + ' H' + data.head
            : 'L' + data.layer + ' MLP';
        label.textContent = 'Steering ' + target + ' x' + data.scale.toFixed(1);
        container.appendChild(label);

        // Side-by-side comparison
        var sideBySide = document.createElement('div');
        sideBySide.className = 'side-by-side';

        renderPredictionPanel(sideBySide, 'Baseline (scale=1.0)', data.baseline_predictions, 'var(--success)');
        renderPredictionPanel(sideBySide, 'Steered (scale=' + data.scale.toFixed(1) + ')', data.steered_predictions, 'var(--highlight)');

        container.appendChild(sideBySide);
    }

    // =======================================================================
    // 3. ACTIVATION SWAPPING
    // =======================================================================
    function buildSwappingSection(parent) {
        var body = createSubsection(
            parent,
            'circuits-swapping',
            'Activation Swapping',
            'Feature Transplant',
            'Swap activations from a source prompt into a target prompt\'s forward pass ' +
            'at a specific layer. Demonstrates that representations encode transferable ' +
            'semantic content (e.g., swapping "France" for "Germany" changes the predicted capital).'
        );

        // Dual prompt inputs
        var dualGroup = document.createElement('div');
        dualGroup.className = 'dual-prompt-group';

        var sourceCol = document.createElement('div');
        sourceCol.className = 'prompt-col';
        sourceCol.innerHTML =
            '<label>Source Prompt</label>' +
            '<input type="text" id="swapping-source-input" placeholder="He was born in London in 1900">';

        var targetCol = document.createElement('div');
        targetCol.className = 'prompt-col';
        targetCol.innerHTML =
            '<label>Target Prompt</label>' +
            '<input type="text" id="swapping-target-input" placeholder="He was born in Berlin in 1900">';

        dualGroup.appendChild(sourceCol);
        dualGroup.appendChild(targetCol);
        body.appendChild(dualGroup);

        // Controls row
        var controls = document.createElement('div');
        controls.className = 'circuits-controls-row';

        // Layer dropdown
        var layerGroup = document.createElement('div');
        layerGroup.className = 'select-group';
        layerGroup.innerHTML = '<label>Swap at layer:</label>';
        var layerSelect = document.createElement('select');
        layerSelect.className = 'select-input';
        layerSelect.id = 'swapping-layer';
        for (var i = 0; i < 12; i++) {
            var opt = document.createElement('option');
            opt.value = i;
            opt.textContent = 'Layer ' + i;
            layerSelect.appendChild(opt);
        }
        layerGroup.appendChild(layerSelect);
        controls.appendChild(layerGroup);

        // Component dropdown
        var compGroup = document.createElement('div');
        compGroup.className = 'select-group';
        compGroup.innerHTML = '<label>Component:</label>';
        var compSelect = document.createElement('select');
        compSelect.className = 'select-input';
        compSelect.id = 'swapping-component';
        [
            { value: 'residual', label: 'Full Residual' },
            { value: 'attn', label: 'Attention Only' },
            { value: 'mlp', label: 'MLP Only' },
        ].forEach(function (opt) {
            var o = document.createElement('option');
            o.value = opt.value;
            o.textContent = opt.label;
            compSelect.appendChild(o);
        });
        compGroup.appendChild(compSelect);
        controls.appendChild(compGroup);

        var btn = document.createElement('button');
        btn.className = 'btn btn-primary';
        btn.id = 'swapping-run-btn';
        btn.textContent = 'Swap';
        controls.appendChild(btn);

        body.appendChild(controls);

        // Results
        var results = document.createElement('div');
        results.className = 'circuits-results';
        results.id = 'swapping-results';
        body.appendChild(results);

        btn.addEventListener('click', function () {
            handleSwapping(btn, results);
        });

        ['swapping-source-input', 'swapping-target-input'].forEach(function (id) {
            document.getElementById(id).addEventListener('keydown', function (e) {
                if (e.key === 'Enter') handleSwapping(btn, results);
            });
        });
    }

    async function handleSwapping(btn, resultsDiv) {
        var sourcePrompt = document.getElementById('swapping-source-input').value.trim();
        var targetPrompt = document.getElementById('swapping-target-input').value.trim();
        if (!sourcePrompt || !targetPrompt) return;

        var layer = parseInt(document.getElementById('swapping-layer').value);
        var component = document.getElementById('swapping-component').value;

        btn.disabled = true;
        btn.textContent = 'Swapping...';
        resultsDiv.innerHTML = '';
        showInlineLoading('swapping-results');

        try {
            var data = await apiFetch('/api/circuits/swapping', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    source_prompt: sourcePrompt,
                    target_prompt: targetPrompt,
                    layer: layer,
                    component: component,
                }),
            });

            resultsDiv.innerHTML = '';
            renderSwappingResults(resultsDiv, data);
        } catch (err) {
            resultsDiv.innerHTML = '';
            showError('swapping-results', 'Swapping failed: ' + err.message);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Swap';
        }
    }

    function renderSwappingResults(container, data) {
        // KL badge
        container.appendChild(createKLBadge(data.kl_divergence || 0));

        // Label
        var label = document.createElement('div');
        label.className = 'circuits-subsection-desc';
        label.textContent = 'Swapped ' + data.component + ' at Layer ' + data.layer +
            ' from source into target';
        container.appendChild(label);

        // Side-by-side comparison
        var sideBySide = document.createElement('div');
        sideBySide.className = 'side-by-side';

        renderPredictionPanel(sideBySide, 'Target Baseline', data.baseline_predictions, 'var(--success)');
        renderPredictionPanel(sideBySide, 'After Swapping', data.swapped_predictions, 'var(--highlight)');

        container.appendChild(sideBySide);
    }

    // =======================================================================
    // 4. PRE-COMPUTATION DETECTION
    // =======================================================================
    function buildPrecomputationSection(parent) {
        var body = createSubsection(
            parent,
            'circuits-precomputation',
            'Pre-computation Detection',
            'Planning Ahead',
            'Detect where the model "plans ahead" by checking if future tokens ' +
            '(+2 to +5 positions) appear in intermediate layer predictions before ' +
            'they\'re needed. Based on the forward/backward planning discovery in ' +
            'Anthropic\'s poetry generation analysis.'
        );

        // Prompt input
        var promptGroup = document.createElement('div');
        promptGroup.className = 'prompt-group';
        promptGroup.style.marginBottom = '12px';
        promptGroup.innerHTML =
            '<input type="text" class="prompt-input" id="precomp-prompt-input" ' +
            'placeholder="The Eiffel Tower is located in the city of">' +
            '<button class="btn btn-primary" id="precomp-run-btn">Detect</button>';
        body.appendChild(promptGroup);

        // Results
        var results = document.createElement('div');
        results.className = 'circuits-results';
        results.id = 'precomp-results';
        body.appendChild(results);

        document.getElementById('precomp-run-btn').addEventListener('click', function () {
            handlePrecomputation(this, results);
        });

        document.getElementById('precomp-prompt-input').addEventListener('keydown', function (e) {
            if (e.key === 'Enter') handlePrecomputation(
                document.getElementById('precomp-run-btn'), results
            );
        });
    }

    async function handlePrecomputation(btn, resultsDiv) {
        var prompt = document.getElementById('precomp-prompt-input').value.trim();
        if (!prompt) return;

        btn.disabled = true;
        btn.textContent = 'Scanning layers...';
        resultsDiv.innerHTML = '';
        showInlineLoading('precomp-results');

        try {
            var data = await apiFetch('/api/circuits/precomputation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: prompt, top_k: 5 }),
            });

            resultsDiv.innerHTML = '';
            renderPrecomputationResults(resultsDiv, data);
        } catch (err) {
            resultsDiv.innerHTML = '';
            showError('precomp-results', 'Detection failed: ' + err.message);
        } finally {
            btn.disabled = false;
            btn.textContent = 'Detect';
        }
    }

    function renderPrecomputationResults(container, data) {
        var tokens = data.tokens || [];
        var matrix = data.precomputation_matrix || [];
        var findings = data.findings || [];
        var offsets = data.future_offsets || [2, 3, 4, 5];

        // Findings list
        if (findings.length > 0) {
            var findingsCard = document.createElement('div');
            findingsCard.style.marginBottom = '16px';

            var findingsTitle = document.createElement('div');
            findingsTitle.className = 'circuits-subsection-desc';
            findingsTitle.innerHTML = '<b>' + findings.length + ' pre-computation instance(s) found:</b>';
            findingsCard.appendChild(findingsTitle);

            var list = document.createElement('ul');
            list.className = 'findings-list';

            findings.slice(0, 10).forEach(function (f) {
                var li = document.createElement('li');

                var badge = document.createElement('span');
                badge.className = 'finding-depth-badge';
                if (f.first_depth_idx <= 3) badge.className += ' early';
                else if (f.first_depth_idx <= 7) badge.className += ' mid';
                else badge.className += ' late';
                badge.textContent = f.first_depth === 'embedding' ? 'Embed' : 'L' + f.first_depth;
                li.appendChild(badge);

                var text = document.createElement('span');
                text.innerHTML =
                    'At position "' + escapeHtml(f.token) + '", future token "' +
                    '<b>' + escapeHtml(f.future_token) + '</b>" (+' + f.future_offset +
                    ') first appears in top-5 predictions';
                li.appendChild(text);

                list.appendChild(li);
            });

            findingsCard.appendChild(list);
            container.appendChild(findingsCard);
        } else {
            var noFindings = document.createElement('div');
            noFindings.className = 'summary-text';
            noFindings.style.marginBottom = '16px';
            noFindings.textContent = 'No notable pre-computation detected. ' +
                'Try longer or more structured prompts (e.g., factual statements, lists).';
            container.appendChild(noFindings);
        }

        // Heatmap: positions x future offsets, colored by first-appearance depth
        if (matrix.length > 0 && tokens.length > 0) {
            var chartDiv = document.createElement('div');
            chartDiv.id = 'precomp-heatmap';
            chartDiv.style.width = '100%';
            chartDiv.style.height = Math.max(300, tokens.length * 28 + 80) + 'px';
            container.appendChild(chartDiv);

            // Build z-data: matrix[pos][offset_idx], null -> 13 (never found)
            var numDepths = 13; // embedding + 12 layers
            var zData = [];
            var hoverText = [];
            var yLabels = [];

            for (var pos = 0; pos < matrix.length; pos++) {
                var row = [];
                var hoverRow = [];
                yLabels.push(tokens[pos]);
                for (var oi = 0; oi < offsets.length; oi++) {
                    var val = matrix[pos][oi];
                    if (val === null || val === undefined) {
                        row.push(numDepths);
                        hoverRow.push('Not found in top-5');
                    } else {
                        row.push(val);
                        var depthLabel = val === 0 ? 'Embedding' : 'Layer ' + (val - 1);
                        hoverRow.push('First at: ' + depthLabel);
                    }
                }
                zData.push(row);
                hoverText.push(hoverRow);
            }

            var xLabels = offsets.map(function (o) { return '+' + o; });

            var traces = [{
                z: zData,
                x: xLabels,
                y: yLabels,
                text: hoverText,
                type: 'heatmap',
                colorscale: [
                    [0, '#e94560'],
                    [0.25, '#f0a500'],
                    [0.5, '#4ecca3'],
                    [0.75, '#16213e'],
                    [1, '#0a0a1a'],
                ],
                zmin: 0,
                zmax: numDepths,
                hovertemplate: 'Position: %{y}<br>Future offset: %{x}<br>%{text}<extra></extra>',
                colorbar: {
                    title: { text: 'First Depth', font: { color: '#e0e0e0', size: 12 } },
                    tickvals: [0, 3, 6, 9, 12, 13],
                    ticktext: ['Embed', 'L2', 'L5', 'L8', 'L11', 'Never'],
                    tickfont: { color: '#e0e0e0' },
                },
            }];

            var layout = darkLayout({
                title: {
                    text: 'Pre-computation: Earliest Layer Where Future Token Appears in Top-5',
                    font: { size: 13, color: '#8892a4' },
                    x: 0.5,
                },
                xaxis: { title: 'Future Token Offset', gridcolor: '#2a2a4a' },
                yaxis: {
                    title: 'Position',
                    autorange: 'reversed',
                    gridcolor: '#2a2a4a',
                    tickfont: { family: 'Courier New', size: 11 },
                },
                margin: { t: 50, r: 120, b: 50, l: 120 },
            });

            Plotly.newPlot(chartDiv, traces, layout, window.PLOTLY_CONFIG);
        }
    }
})();
