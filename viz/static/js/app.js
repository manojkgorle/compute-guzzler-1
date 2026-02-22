/**
 * app.js — Tab router, WebSocket init, and shared infrastructure.
 */

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
window.vizState = {
    socket: null,
    model_info: null,
    activeTab: 'dashboard',
    tabInitialized: {
        dashboard: false,
        attention: false,
        activations: false,
        attribution: false,
        circuits: false,
    },
};

// ---------------------------------------------------------------------------
// Shared Plotly layout / config
// ---------------------------------------------------------------------------
window.PLOTLY_DARK_LAYOUT = {
    paper_bgcolor: '#1a1a2e',
    plot_bgcolor: '#16213e',
    font: { color: '#e0e0e0', family: '-apple-system, BlinkMacSystemFont, sans-serif' },
    margin: { t: 36, r: 24, b: 48, l: 56 },
    xaxis: {
        gridcolor: '#2a2a4a',
        zerolinecolor: '#2a2a4a',
    },
    yaxis: {
        gridcolor: '#2a2a4a',
        zerolinecolor: '#2a2a4a',
    },
    legend: {
        bgcolor: 'rgba(22,33,62,0.8)',
        bordercolor: '#2a2a4a',
        borderwidth: 1,
        font: { size: 11 },
    },
};

window.PLOTLY_CONFIG = { responsive: true, displayModeBar: false };

// ---------------------------------------------------------------------------
// Tab routing
// ---------------------------------------------------------------------------
function switchTab(tabName) {
    // Hide all tab content
    document.querySelectorAll('.tab-content').forEach(function (el) {
        el.classList.remove('active');
    });

    // Deactivate all nav buttons
    document.querySelectorAll('.nav-btn').forEach(function (btn) {
        btn.classList.remove('active');
    });

    // Show target tab content
    var section = document.getElementById('tab-' + tabName);
    if (section) {
        section.classList.add('active');
    }

    // Activate corresponding nav button
    var btn = document.querySelector('.nav-btn[data-tab="' + tabName + '"]');
    if (btn) {
        btn.classList.add('active');
    }

    window.vizState.activeTab = tabName;

    // Lazy-init the tab if first visit
    if (!window.vizState.tabInitialized[tabName]) {
        window.vizState.tabInitialized[tabName] = true;
        switch (tabName) {
            case 'dashboard':
                if (typeof initDashboard === 'function') initDashboard();
                break;
            case 'attention':
                if (typeof initAttention === 'function') initAttention();
                break;
            case 'activations':
                if (typeof initActivations === 'function') initActivations();
                break;
            case 'attribution':
                if (typeof initAttribution === 'function') initAttribution();
                break;
            case 'circuits':
                if (typeof initCircuits === 'function') initCircuits();
                break;
        }
    }

    // Trigger Plotly relayout so charts resize correctly when tab becomes visible
    setTimeout(function () {
        section.querySelectorAll('.js-plotly-plot').forEach(function (plot) {
            Plotly.Plots.resize(plot);
        });
    }, 50);
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------
function initWebSocket() {
    try {
        var socket = io({ transports: ['polling'], upgrade: false });

        socket.on('connect', function () {
            console.log('[ws] connected');
        });

        socket.on('disconnect', function () {
            console.log('[ws] disconnected');
        });

        socket.on('connect_error', function (err) {
            console.warn('[ws] connection error:', err.message);
        });

        // Dispatch domain events to tab handlers
        socket.on('step_update', function (data) {
            if (typeof handleStepUpdate === 'function') handleStepUpdate(data);
        });

        socket.on('val_update', function (data) {
            if (typeof handleValUpdate === 'function') handleValUpdate(data);
        });

        window.vizState.socket = socket;
    } catch (err) {
        console.warn('[ws] failed to initialize:', err.message);
    }
}

// ---------------------------------------------------------------------------
// Loading helpers
// ---------------------------------------------------------------------------
function showLoading(containerId) {
    var container = document.getElementById(containerId);
    if (!container) return;

    // Avoid duplicate overlays
    if (container.querySelector('.loading-overlay')) return;

    container.style.position = 'relative';
    var overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.innerHTML = '<div class="spinner"></div>';
    container.appendChild(overlay);
}

function hideLoading(containerId) {
    var container = document.getElementById(containerId);
    if (!container) return;
    var overlay = container.querySelector('.loading-overlay');
    if (overlay) overlay.remove();
}

function showInlineLoading(containerId) {
    var container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML =
        '<div class="loading-inline"><div class="spinner"></div><span>Loading...</span></div>';
}

function hideInlineLoading(containerId) {
    var container = document.getElementById(containerId);
    if (!container) return;
    var loader = container.querySelector('.loading-inline');
    if (loader) loader.remove();
}

// ---------------------------------------------------------------------------
// Error display
// ---------------------------------------------------------------------------
function showError(containerId, message) {
    var container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML =
        '<div class="error-message">' + escapeHtml(message) + '</div>';
}

function escapeHtml(text) {
    var div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ---------------------------------------------------------------------------
// Reusable prompt input component
// ---------------------------------------------------------------------------
function createPromptInput(containerId, onSubmit, options) {
    var opts = options || {};
    var placeholder = opts.placeholder || 'Enter a prompt...';
    var btnLabel = opts.buttonLabel || 'Analyze';

    var container = document.getElementById(containerId);
    if (!container) return null;

    var group = document.createElement('div');
    group.className = 'prompt-group';

    var input = document.createElement('input');
    input.type = 'text';
    input.className = 'prompt-input';
    input.placeholder = placeholder;
    input.id = containerId + '-prompt-input';

    var btn = document.createElement('button');
    btn.className = 'btn btn-primary';
    btn.textContent = btnLabel;
    btn.id = containerId + '-prompt-btn';

    group.appendChild(input);
    group.appendChild(btn);
    container.appendChild(group);

    function submit() {
        var value = input.value.trim();
        if (!value) return;
        onSubmit(value);
    }

    btn.addEventListener('click', submit);
    input.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') submit();
    });

    return { input: input, button: btn, group: group };
}

// ---------------------------------------------------------------------------
// Helpers: create a select dropdown inside a container
// ---------------------------------------------------------------------------
function createDropdown(containerId, label, id, optionsList) {
    var container = document.getElementById(containerId);
    if (!container) return null;

    var group = document.createElement('div');
    group.className = 'select-group';

    var lbl = document.createElement('label');
    lbl.setAttribute('for', id);
    lbl.textContent = label;

    var sel = document.createElement('select');
    sel.className = 'select-input';
    sel.id = id;

    optionsList.forEach(function (opt) {
        var o = document.createElement('option');
        o.value = opt.value;
        o.textContent = opt.label;
        sel.appendChild(o);
    });

    group.appendChild(lbl);
    group.appendChild(sel);
    container.appendChild(group);

    return sel;
}

// ---------------------------------------------------------------------------
// Fetch helper with JSON error handling
// ---------------------------------------------------------------------------
async function apiFetch(url, options) {
    var resp = await fetch(url, options);
    if (!resp.ok) {
        var errText = await resp.text();
        throw new Error('API error ' + resp.status + ': ' + errText);
    }
    return resp.json();
}

// ---------------------------------------------------------------------------
// Deep-merge Plotly layout with dark defaults
// ---------------------------------------------------------------------------
function darkLayout(overrides) {
    var base = JSON.parse(JSON.stringify(window.PLOTLY_DARK_LAYOUT));
    return Object.assign(base, overrides || {});
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', function () {
    // Register nav click handlers
    document.querySelectorAll('.nav-btn').forEach(function (btn) {
        btn.addEventListener('click', function () {
            switchTab(btn.dataset.tab);
        });
    });

    // Init WebSocket
    initWebSocket();

    // Fetch model info
    apiFetch('/api/model/info')
        .then(function (data) {
            window.vizState.model_info = data;
            console.log('[app] model info loaded:', data);
        })
        .catch(function (err) {
            console.warn('[app] could not fetch model info:', err.message);
        });

    // Activate first tab
    switchTab('dashboard');
});
