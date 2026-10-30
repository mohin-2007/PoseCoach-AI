// ==================================================
// POSECOACH AI - HIGH-SPEED CLIENT ENGINE
// Ultra-Responsive Posture Tracking, Biomechanics, Vision & Voice
// ==================================================

// ==================================================
// 1. CONFIGURATION & SPEED PROFILES
// ==================================================

const SPEED_CONFIGS = {
    turbo: {
        interval: 80,       // ms between frame processing
        maxWidth: 380,      // downsampled capture width for <2ms encoding
        jpegQuality: 0.65,  // lightweight ~12KB payload
        voteWindow: 3,      // 3-frame evaluation window
        requiredVotes: 2    // 2 matching frames to confirm
    },
    balanced: {
        interval: 140,
        maxWidth: 440,
        jpegQuality: 0.70,
        voteWindow: 3,
        requiredVotes: 2
    },
    precision: {
        interval: 220,
        maxWidth: 540,
        jpegQuality: 0.75,
        voteWindow: 4,
        requiredVotes: 3
    }
};

let activeSpeedMode = "turbo";
let MIN_FRAME_CONFIDENCE = 45.0;
let voiceSensitivityThreshold = 50.0;

// MediaPipe 33 Landmark Body Connection Topology
const POSE_CONNECTIONS = [
    // Torso
    [11, 12], [11, 23], [12, 24], [23, 24],
    // Left Arm
    [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
    // Right Arm
    [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
    // Left Leg
    [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
    // Right Leg
    [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
    // Head / Face
    [0, 1], [1, 2], [2, 3], [3, 7],
    [0, 4], [4, 5], [5, 6], [6, 8],
    [9, 10]
];

const POSTURE_GLYPHS = {
    Standing: `<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="4" r="2.5"></circle><line x1="12" y1="6.5" x2="12" y2="15"></line><line x1="8" y1="10" x2="16" y2="10"></line><line x1="12" y1="15" x2="9" y2="22"></line><line x1="12" y1="15" x2="15" y2="22"></line></svg>`,
    Sitting: `<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="9" cy="4" r="2.5"></circle><path d="M9 7v6h7v7"></path><path d="M9 13H5"></path></svg>`,
    Squatting: `<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="5" r="2.5"></circle><path d="M12 7.5v4l-4 3 2 5"></path><path d="M12 11.5l4 3-2 5"></path></svg>`,
    Bending: `<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="6" cy="11" r="2.5"></circle><path d="M8 12h5l3 8"></path><path d="M13 12l2-4"></path></svg>`,
    None: `<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="5" r="3"></circle><path d="M6 10h12M12 13v8"></path></svg>`
};

// Global set to retain utterances and prevent browser V8 garbage collection bug
window.activeSpeechUtterances = window.activeSpeechUtterances || new Set();

// Web Audio Context for zero-latency audio chimes
let audioCtx = null;
function getAudioContext() {
    if (!audioCtx) {
        const AudioClass = window.AudioContext || window.webkitAudioContext;
        if (AudioClass) {
            audioCtx = new AudioClass();
        }
    }
    if (audioCtx && audioCtx.state === "suspended") {
        audioCtx.resume();
    }
    return audioCtx;
}

// Play pleasant synthetic two-tone chime
function playChime(type = "transition") {
    const audioToggle = document.getElementById("audioChimeToggle");
    if (audioToggle && !audioToggle.checked) return;

    try {
        const ctx = getAudioContext();
        if (!ctx) return;

        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);

        const now = ctx.currentTime;
        if (type === "rep") {
            // High celebratory ping for squat rep
            osc.frequency.setValueAtTime(659.25, now);       // E5
            osc.frequency.exponentialRampToValueAtTime(880.0, now + 0.08); // A5
            gain.gain.setValueAtTime(0.15, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.16);
            osc.start(now);
            osc.stop(now + 0.16);
        } else {
            // Gentle transition chime
            osc.frequency.setValueAtTime(523.25, now);       // C5
            osc.frequency.exponentialRampToValueAtTime(659.25, now + 0.06); // E5
            gain.gain.setValueAtTime(0.12, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.14);
            osc.start(now);
            osc.stop(now + 0.14);
        }
    } catch (e) {
        // AudioContext not allowed or not supported yet
    }
}

// ==================================================
// 2. DOM ELEMENT REFERENCES
// ==================================================

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const skeletonOverlay = document.getElementById("skeletonOverlay");
const cameraStage = document.getElementById("cameraStage");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const startBtnInPlaceholder = document.getElementById("startBtnInPlaceholder");
const skeletonToggleBtn = document.getElementById("skeletonToggleBtn");
const skeletonStatusText = document.getElementById("skeletonStatusText");
const speedModeSelect = document.getElementById("speedModeSelect");

const cameraStatusBadge = document.getElementById("cameraStatusBadge");
const frameStatusText = document.getElementById("frameStatusText");
const cameraMessage = document.getElementById("cameraMessage");
const liveFps = document.getElementById("liveFps");
const liveLatency = document.getElementById("liveLatency");
const hudVisibility = document.getElementById("hudVisibility");

// Biomechanics & Squat Rep Elements
const biomechSpine = document.getElementById("biomechSpine");
const biomechKnee = document.getElementById("biomechKnee");
const biomechForm = document.getElementById("biomechForm");
const hudSquatCount = document.getElementById("hudSquatCount");
const dashSquatsCount = document.getElementById("dashSquatsCount");
const resetSquatBtn = document.getElementById("resetSquatBtn");

const postureName = document.getElementById("postureName");
const postureGlyph = document.getElementById("postureGlyph");
const predictionMessage = document.getElementById("predictionMessage");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const voteWindowProgress = document.getElementById("voteWindowProgress");
const voiceThresholdHint = document.getElementById("voiceThresholdHint");

// Session Tracker Elements
const sessionStartBtn = document.getElementById("sessionStartBtn");
const sessionPauseBtn = document.getElementById("sessionPauseBtn");
const sessionResumeBtn = document.getElementById("sessionResumeBtn");
const sessionStopBtn = document.getElementById("sessionStopBtn");
const quickStartSessionBtn = document.getElementById("quickStartSessionBtn");
const quickSessionBtnText = document.getElementById("quickSessionBtnText");

const trackerSessionClock = document.getElementById("trackerSessionClock");
const trackerStatusChip = document.getElementById("trackerStatusChip");
const trackerActivePosture = document.getElementById("trackerActivePosture");
const trackerCurrentPostureDuration = document.getElementById("trackerCurrentPostureDuration");

const trackStandClock = document.getElementById("trackStandClock");
const trackSitClock = document.getElementById("trackSitClock");
const trackSquatClock = document.getElementById("trackSquatClock");
const trackBendClock = document.getElementById("trackBendClock");
const timelineList = document.getElementById("timelineList");
const timelineEmptyState = document.getElementById("timelineEmptyState");
const timelineCount = document.getElementById("timelineCount");

// Dashboard Elements
const dashTotalTime = document.getElementById("dashTotalTime");
const dashSessionState = document.getElementById("dashSessionState");
const dashCurrentPosture = document.getElementById("dashCurrentPosture");
const dashPostureTimer = document.getElementById("dashPostureTimer");
const dashTransitions = document.getElementById("dashTransitions");
const dashStandingTime = document.getElementById("dashStandingTime");
const dashSittingTime = document.getElementById("dashSittingTime");
const dashSquattingTime = document.getElementById("dashSquattingTime");
const dashBendingTime = document.getElementById("dashBendingTime");
const dashStandingBar = document.getElementById("dashStandingBar");
const dashSittingBar = document.getElementById("dashSittingBar");
const dashSquattingBar = document.getElementById("dashSquattingBar");
const dashBendingBar = document.getElementById("dashBendingBar");
const dashStandingPct = document.getElementById("dashStandingPct");
const dashSittingPct = document.getElementById("dashSittingPct");
const dashSquattingPct = document.getElementById("dashSquattingPct");
const dashBendingPct = document.getElementById("dashBendingPct");
const dashLongestSit = document.getElementById("dashLongestSit");
const dashLongestBend = document.getElementById("dashLongestBend");
const dashSittingThreshold = document.getElementById("dashSittingThreshold");
const dashBendingThreshold = document.getElementById("dashBendingThreshold");
const dashReminderStatus = document.getElementById("dashReminderStatus");

// Nav indicator
const navSessionDot = document.getElementById("navSessionDot");
const navSessionText = document.getElementById("navSessionText");

// Voice Elements
const voiceToggle = document.getElementById("voiceToggle");
const voiceSelect = document.getElementById("voiceSelect");
const speechRate = document.getElementById("speechRate");
const rateValue = document.getElementById("rateValue");
const voiceStatus = document.getElementById("voiceStatus");
const testVoiceBtn = document.getElementById("testVoiceBtn");
const voiceSensitivitySlider = document.getElementById("voiceSensitivitySlider");
const voiceSensitivityVal = document.getElementById("voiceSensitivityVal");

// Contexts
const captureCtx = canvas.getContext("2d", { willReadFrequently: true });
const skeletonCtx = skeletonOverlay.getContext("2d");

// ==================================================
// 3. APPLICATION STATE
// ==================================================

let cameraStream = null;
let cameraRunning = false;
let processingFrame = false;
let showSkeletonOverlay = true;

// Performance & FPS tracking
let lastFrameTime = performance.now();
let frameCount = 0;
let currentFps = 0;
let rollingLatency = 0;

// ML prediction voting window
let voteHistory = [];
let confirmedPosture = null;
let lastAnnouncedPosture = null;
let lastSpokenTimestamp = 0;

// Squat Repetition Counter State
let squatRepCount = 0;
let squatFsmStage = "up"; // 'up', 'down'

// Posture Tracking & Session State
let sessionState = "idle"; // 'idle', 'running', 'paused'
let activeSessionId = null;
let sessionStartTime = null;
let lastTickTime = null;

// Durations in seconds (float)
let totalDurationSeconds = 0.0;
let standingSeconds = 0.0;
let sittingSeconds = 0.0;
let squattingSeconds = 0.0;
let bendingSeconds = 0.0;

// Current posture interval tracking
let currentIntervalPosture = null;
let currentIntervalStartTime = null;
let currentIntervalDuration = 0.0;
let currentIntervalConfidences = [];
let totalTransitionsCount = 0;
let longestSittingSeconds = 0.0;
let longestBendingSeconds = 0.0;

// Reminder alerts flags
let sittingAlertTriggered = false;
let bendingAlertTriggered = false;

// User settings
let userSettings = {
    sitting_reminder_minutes: 30,
    bending_reminder_minutes: 10,
    reminder_notifications_enabled: 1,
    reminder_voice_enabled: 1,
    voice_enabled: 1,
    speech_rate: 1.0,
    selected_voice: "",
    preferred_language: "en"
};

// Chart.js instances
let dashChartInstance = null;
let analyticsPieInstance = null;
let analyticsBarInstance = null;
let analyticsTrendInstance = null;

// ==================================================
// 4. TAB NAVIGATION
// ==================================================

function switchTab(tabId) {
    document.querySelectorAll(".nav-tab").forEach(tab => {
        const isActive = tab.getAttribute("data-tab") === tabId;
        tab.classList.toggle("active", isActive);
        tab.setAttribute("aria-selected", isActive);
    });

    document.querySelectorAll(".tab-view").forEach(view => {
        const isActive = view.id === `view-${tabId}`;
        view.classList.toggle("active", isActive);
    });

    if (tabId === "analytics") {
        loadAnalyticsData();
    } else if (tabId === "history") {
        loadHistoryTable();
    } else if (tabId === "dashboard") {
        updateDashboardCharts();
    }
}

document.querySelectorAll(".nav-tab").forEach(tab => {
    tab.addEventListener("click", () => {
        const tabId = tab.getAttribute("data-tab");
        switchTab(tabId);
    });
});

// Speed Mode Selector
if (speedModeSelect) {
    speedModeSelect.addEventListener("change", () => {
        activeSpeedMode = speedModeSelect.value;
        const cfg = SPEED_CONFIGS[activeSpeedMode];
        voteWindowProgress.textContent = `Votes: 0/${cfg.voteWindow}`;
        showToast("Speed Mode Updated", `Switched to ${activeSpeedMode.toUpperCase()} mode (${cfg.interval}ms interval).`, "info");
    });
}

// Reset Squat Reps
if (resetSquatBtn) {
    resetSquatBtn.addEventListener("click", () => {
        squatRepCount = 0;
        squatFsmStage = "up";
        if (hudSquatCount) hudSquatCount.textContent = "0";
        if (dashSquatsCount) dashSquatsCount.textContent = "0";
        showToast("Squats Reset", "Squat counter reset to 0.", "info");
    });
}

// ==================================================
// 5. TOAST NOTIFICATIONS & ALERTS
// ==================================================

function showToast(title, message, type = "info") {
    const container = document.getElementById("toastContainer");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <div class="toast-content">
            <div class="toast-title">${escapeHtml(title)}</div>
            <div class="toast-msg">${escapeHtml(message)}</div>
        </div>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateX(20px)";
        setTimeout(() => toast.remove(), 250);
    }, 4000);
}

function escapeHtml(text) {
    if (!text) return "";
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// Request Browser Web Notification Permission
async function requestNotificationPermission() {
    if (!("Notification" in window)) {
        showToast("Not Supported", "Browser notifications are not supported in this browser.", "warning");
        return;
    }
    const perm = await Notification.requestPermission();
    if (perm === "granted") {
        showToast("Permission Granted", "PoseCoach AI will now deliver native desktop posture reminders.", "success");
    } else {
        showToast("Permission Denied", "Native notifications blocked. In-app alerts will be used.", "warning");
    }
}

function triggerBrowserNotification(title, body) {
    if ("Notification" in window && Notification.permission === "granted") {
        try {
            new Notification(title, {
                body: body,
                icon: "/static/favicon.ico"
            });
        } catch (e) {
            console.error("Browser notification failed:", e);
        }
    }
}

// ==================================================
// 6. ROBUST TEXT-TO-SPEECH VOICE SYSTEM (EN & HI)
// ==================================================

const speechSupported = "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;

if (!speechSupported) {
    if (voiceToggle) {
        voiceToggle.checked = false;
        voiceToggle.disabled = true;
    }
    if (voiceStatus) {
        voiceStatus.textContent = "Text-to-speech not supported in this browser.";
    }
}

function loadVoices() {
    if (!speechSupported) return;

    const voices = window.speechSynthesis.getVoices();
    if (!voices || voices.length === 0) return;

    const currentVal = voiceSelect.value;
    voiceSelect.innerHTML = '<option value="">Default browser voice</option>';

    // Sort: English and Hindi first
    voices.forEach(voice => {
        const lang = voice.lang.toLowerCase();
        if (lang.startsWith("en") || lang.startsWith("hi")) {
            const option = document.createElement("option");
            option.value = voice.voiceURI;
            const langLabel = lang.startsWith("hi") ? "Hindi" : "English";
            option.textContent = `${voice.name} (${langLabel} - ${voice.lang})`;
            voiceSelect.appendChild(option);
        }
    });

    if (currentVal && Array.from(voiceSelect.options).some(o => o.value === currentVal)) {
        voiceSelect.value = currentVal;
    } else if (userSettings.selected_voice) {
        voiceSelect.value = userSettings.selected_voice;
    }
}

if (speechSupported) {
    loadVoices();
    window.speechSynthesis.addEventListener("voiceschanged", loadVoices);
}

// Unlock audio context and TTS on first user interaction
function unlockAudioEngine() {
    getAudioContext();
    if (speechSupported && window.speechSynthesis.paused) {
        window.speechSynthesis.resume();
    }
}

// Reliable speech dispatcher with garbage collection protection & cooldown
function speakPhrase(phrase, force = false) {
    if (!speechSupported || !voiceToggle.checked) return;

    const now = performance.now();
    // 0.8s cooldown prevents overlapping speech
    if (!force && (now - lastSpokenTimestamp < 800)) {
        return;
    }
    lastSpokenTimestamp = now;

    try {
        // Ensure synthesis queue is active
        if (window.speechSynthesis.paused) {
            window.speechSynthesis.resume();
        }

        const utterance = new SpeechSynthesisUtterance(phrase);
        utterance.rate = Number(speechRate.value) || 1.0;

        const selectedURI = voiceSelect.value;
        if (selectedURI) {
            const voices = window.speechSynthesis.getVoices();
            const voice = voices.find(v => v.voiceURI === selectedURI);
            if (voice) utterance.voice = voice;
        }

        // Retain in active set so V8 GC does not garbage collect it mid-speech
        window.activeSpeechUtterances.add(utterance);

        utterance.onend = () => {
            window.activeSpeechUtterances.delete(utterance);
        };
        utterance.onerror = (e) => {
            window.activeSpeechUtterances.delete(utterance);
            console.warn("TTS utterance error:", e);
        };

        window.speechSynthesis.speak(utterance);
    } catch (err) {
        console.error("Speech error:", err);
    }
}

function speakPosture(posture) {
    speakPhrase(posture);
    playChime("transition");
}

// Test Voice Button
if (testVoiceBtn) {
    testVoiceBtn.addEventListener("click", () => {
        unlockAudioEngine();
        speakPhrase("PoseCoach AI voice system is active", true);
        playChime("rep");
        showToast("Voice Test", "Playing speech test audio.", "info");
    });
}

// Voice Sensitivity Slider
if (voiceSensitivitySlider) {
    voiceSensitivitySlider.addEventListener("input", () => {
        voiceSensitivityThreshold = Number(voiceSensitivitySlider.value);
        voiceSensitivityVal.textContent = `${voiceSensitivityThreshold}%`;
        if (voiceThresholdHint) {
            voiceThresholdHint.textContent = `Voice threshold: ${voiceSensitivityThreshold}%`;
        }
    });
}

// Voice controls event listeners
if (speechRate) {
    speechRate.addEventListener("input", () => {
        rateValue.textContent = `${Number(speechRate.value).toFixed(1)}x`;
        userSettings.speech_rate = Number(speechRate.value);
    });
}

if (voiceToggle) {
    voiceToggle.addEventListener("change", () => {
        if (!voiceToggle.checked) {
            window.speechSynthesis.cancel();
            voiceStatus.textContent = "Voice announcements muted.";
        } else {
            voiceStatus.textContent = "Voice announcements enabled.";
            if (confirmedPosture) {
                lastAnnouncedPosture = null;
                speakPosture(confirmedPosture);
            }
        }
    });
}

// ==================================================
// 7. BIOMECHANICS & SKELETON VISUALIZATION
// ==================================================

// Calculate 2D angle between 3 points (A, B, C with B as vertex)
function calculateJointAngle(p1, p2, p3) {
    if (!p1 || !p2 || !p3) return null;
    const v1x = p1.x - p2.x;
    const v1y = p1.y - p2.y;
    const v2x = p3.x - p2.x;
    const v2y = p3.y - p2.y;

    const dot = v1x * v2x + v1y * v2y;
    const mag1 = Math.hypot(v1x, v1y);
    const mag2 = Math.hypot(v2x, v2y);

    if (mag1 < 1e-5 || mag2 < 1e-5) return null;
    const cosAngle = Math.max(-1.0, Math.min(1.0, dot / (mag1 * mag2)));
    return (Math.acos(cosAngle) * 180.0) / Math.PI;
}

// Calculate spine inclination relative to vertical
function calculateSpineInclination(shoulderL, shoulderR, hipL, hipR) {
    if (!shoulderL || !shoulderR || !hipL || !hipR) return null;
    const midShoulderX = (shoulderL.x + shoulderR.x) / 2.0;
    const midShoulderY = (shoulderL.y + shoulderR.y) / 2.0;
    const midHipX = (hipL.x + hipR.x) / 2.0;
    const midHipY = (hipL.y + hipR.y) / 2.0;

    const dx = Math.abs(midShoulderX - midHipX);
    const dy = Math.abs(midShoulderY - midHipY);

    if (dy < 1e-5) return 90.0;
    return (Math.atan2(dx, dy) * 180.0) / Math.PI;
}

// Biomechanical Analysis Evaluator
function evaluateBiomechanics(landmarks) {
    if (!landmarks || landmarks.length < 33) return;

    const sL = landmarks[11], sR = landmarks[12];
    const hL = landmarks[23], hR = landmarks[24];
    const kL = landmarks[25], kR = landmarks[26];
    const aL = landmarks[27], aR = landmarks[28];

    // Spine inclination angle
    const spineAngle = calculateSpineInclination(sL, sR, hL, hR);

    // Knee flexion angles
    const kneeAngleL = calculateJointAngle(hL, kL, aL);
    const kneeAngleR = calculateJointAngle(hR, kR, aR);
    let avgKneeAngle = null;

    if (kneeAngleL !== null && kneeAngleR !== null) {
        avgKneeAngle = (kneeAngleL + kneeAngleR) / 2.0;
    } else {
        avgKneeAngle = kneeAngleL !== null ? kneeAngleL : kneeAngleR;
    }

    // Update HUD readouts
    if (spineAngle !== null) {
        biomechSpine.textContent = `${Math.round(spineAngle)}°`;
        if (spineAngle < 22) {
            biomechSpine.style.color = "var(--emerald-500)";
        } else if (spineAngle < 40) {
            biomechSpine.style.color = "var(--amber-500)";
        } else {
            biomechSpine.style.color = "var(--rose-500)";
        }
    }

    if (avgKneeAngle !== null) {
        biomechKnee.textContent = `${Math.round(avgKneeAngle)}°`;
        if (avgKneeAngle > 140) {
            biomechKnee.style.color = "var(--blue-500)"; // Extended (Standing)
        } else if (avgKneeAngle <= 100) {
            biomechKnee.style.color = "var(--emerald-500)"; // Deep Squat
        } else {
            biomechKnee.style.color = "var(--amber-500)";
        }
    }

    // Determine ergonomic quality label
    if (confirmedPosture === "Standing") {
        if (spineAngle !== null && spineAngle < 22) {
            biomechForm.textContent = "Optimal Spine";
            biomechForm.style.color = "var(--emerald-500)";
        } else {
            biomechForm.textContent = "Forward Lean";
            biomechForm.style.color = "var(--amber-500)";
        }
    } else if (confirmedPosture === "Sitting") {
        if (spineAngle !== null && spineAngle < 26) {
            biomechForm.textContent = "Ergonomic";
            biomechForm.style.color = "var(--emerald-500)";
        } else {
            biomechForm.textContent = "Slouch Warning";
            biomechForm.style.color = "var(--rose-500)";
        }
    } else if (confirmedPosture === "Squatting") {
        if (avgKneeAngle !== null && avgKneeAngle <= 100) {
            biomechForm.textContent = "Deep Squat Depth";
            biomechForm.style.color = "var(--emerald-500)";
        } else {
            biomechForm.textContent = "Partial Squat";
            biomechForm.style.color = "var(--amber-500)";
        }
    } else if (confirmedPosture === "Bending") {
        biomechForm.textContent = "Spine Flexion";
        biomechForm.style.color = "var(--rose-500)";
    }

    // Squat Repetition FSM Logic
    if (avgKneeAngle !== null) {
        if (squatFsmStage === "up" && (confirmedPosture === "Squatting" || avgKneeAngle < 115)) {
            squatFsmStage = "down";
        } else if (squatFsmStage === "down" && (confirmedPosture === "Standing" || avgKneeAngle > 145)) {
            squatFsmStage = "up";
            squatRepCount++;
            if (hudSquatCount) hudSquatCount.textContent = squatRepCount;
            if (dashSquatsCount) dashSquatsCount.textContent = squatRepCount;
            playChime("rep");

            // Voice announce every rep
            speakPhrase(`Rep ${squatRepCount}`);
        }
    }
}

// Draw skeleton overlay
function drawSkeleton(landmarks) {
    if (!showSkeletonOverlay || !landmarks || landmarks.length !== 33) {
        skeletonCtx.clearRect(0, 0, skeletonOverlay.width, skeletonOverlay.height);
        return;
    }

    const width = skeletonOverlay.width;
    const height = skeletonOverlay.height;

    skeletonCtx.clearRect(0, 0, width, height);

    // Glowing cyan connections
    skeletonCtx.lineWidth = 2.5;
    skeletonCtx.strokeStyle = "rgba(56, 189, 248, 0.85)";
    skeletonCtx.shadowColor = "#38bdf8";
    skeletonCtx.shadowBlur = 6;

    for (const [i1, i2] of POSE_CONNECTIONS) {
        const pt1 = landmarks[i1];
        const pt2 = landmarks[i2];

        if (pt1 && pt2 && pt1.visibility >= 0.35 && pt2.visibility >= 0.35) {
            skeletonCtx.beginPath();
            skeletonCtx.moveTo(pt1.x * width, pt1.y * height);
            skeletonCtx.lineTo(pt2.x * width, pt2.y * height);
            skeletonCtx.stroke();
        }
    }

    // Glowing joint dots
    skeletonCtx.shadowBlur = 4;
    for (let i = 0; i < landmarks.length; i++) {
        const pt = landmarks[i];
        if (!pt || pt.visibility < 0.35) continue;

        const x = pt.x * width;
        const y = pt.y * height;

        skeletonCtx.beginPath();
        skeletonCtx.arc(x, y, 3.5, 0, 2 * Math.PI);

        if (i === 0) {
            skeletonCtx.fillStyle = "#38bdf8"; // Nose
        } else if (i === 11 || i === 12 || i === 23 || i === 24) {
            skeletonCtx.fillStyle = "#a855f7"; // Torso anchors (violet)
        } else if (i >= 25) {
            skeletonCtx.fillStyle = "#10b981"; // Legs / knees / feet (emerald)
        } else {
            skeletonCtx.fillStyle = "#38bdf8"; // Arms / shoulders (cyan)
        }

        skeletonCtx.fill();
    }

    skeletonCtx.shadowBlur = 0;
}

if (skeletonToggleBtn) {
    skeletonToggleBtn.addEventListener("click", () => {
        showSkeletonOverlay = !showSkeletonOverlay;
        skeletonStatusText.textContent = showSkeletonOverlay ? "ON" : "OFF";
        if (!showSkeletonOverlay) {
            skeletonCtx.clearRect(0, 0, skeletonOverlay.width, skeletonOverlay.height);
        }
    });
}

// ==================================================
// 8. DURATION FORMATTERS & UTILITIES
// ==================================================

function formatHms(seconds) {
    const s = Math.max(0, Math.floor(seconds));
    const hours = Math.floor(s / 3600);
    const minutes = Math.floor((s % 3600) / 60);
    const secs = s % 60;
    const hh = String(hours).padStart(2, "0");
    const mm = String(minutes).padStart(2, "0");
    const ss = String(secs).padStart(2, "0");
    return `${hh}:${mm}:${ss}`;
}

function formatMinSec(seconds) {
    const s = Math.max(0, Math.floor(seconds));
    const minutes = Math.floor(s / 60);
    const secs = s % 60;
    return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function formatTimeString(isoString) {
    try {
        const d = new Date(isoString);
        return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    } catch {
        return isoString;
    }
}

// ==================================================
// 9. HIGH-SPEED PREDICTION PROCESSING & VOTING
// ==================================================

function resetVoting() {
    voteHistory = [];
    const cfg = SPEED_CONFIGS[activeSpeedMode];
    voteWindowProgress.textContent = `Votes: 0/${cfg.voteWindow}`;
}

function confirmStablePosture() {
    const cfg = SPEED_CONFIGS[activeSpeedMode];
    if (voteHistory.length < cfg.requiredVotes) return;

    const counts = {};
    const totalConf = {};

    voteHistory.forEach(item => {
        counts[item.posture] = (counts[item.posture] || 0) + 1;
        totalConf[item.posture] = (totalConf[item.posture] || 0) + item.confidence;
    });

    let bestPosture = null;
    let maxCount = 0;

    for (const posture in counts) {
        if (counts[posture] > maxCount) {
            maxCount = counts[posture];
            bestPosture = posture;
        }
    }

    if (maxCount >= cfg.requiredVotes && bestPosture) {
        const avgConfidence = totalConf[bestPosture] / maxCount;
        setConfirmedPosture(bestPosture, avgConfidence);
    }
}

function setConfirmedPosture(posture, avgConfidence) {
    const isNewPosture = confirmedPosture !== posture;
    confirmedPosture = posture;

    postureName.textContent = posture;
    postureGlyph.innerHTML = POSTURE_GLYPHS[posture] || POSTURE_GLYPHS.None;
    frameStatusText.textContent = "CONFIRMED";

    // Track active interval if session is running
    if (sessionState === "running" && isNewPosture) {
        onPostureTransition(posture, avgConfidence);
    }

    // Voice announcement: check against user sensitivity threshold
    if (isNewPosture && avgConfidence >= voiceSensitivityThreshold) {
        lastAnnouncedPosture = posture;
        speakPosture(posture);
    }
}

function handlePrediction(data) {
    if (!data.detected) {
        resetVoting();
        postureName.textContent = "No Person Detected";
        postureGlyph.innerHTML = POSTURE_GLYPHS.None;
        predictionMessage.textContent = data.message || "Position full body inside camera frame.";
        confidenceValue.textContent = "0.0%";
        confidenceBar.style.width = "0%";
        frameStatusText.textContent = "SEARCHING";
        hudVisibility.textContent = "VISIBILITY: LOW";

        if (showSkeletonOverlay) {
            skeletonCtx.clearRect(0, 0, skeletonOverlay.width, skeletonOverlay.height);
        }
        return;
    }

    const posture = data.posture;
    const confidence = Number(data.confidence);
    const visibility = Number(data.visibility || 0);

    hudVisibility.textContent = `VISIBILITY: ${(visibility * 100).toFixed(0)}%`;

    // Draw live skeleton & evaluate biomechanical angles
    if (data.landmarks && data.landmarks.length > 0) {
        drawSkeleton(data.landmarks);
        evaluateBiomechanics(data.landmarks);
    }

    // Confidence filter
    if (!POSTURE_GLYPHS[posture] || !Number.isFinite(confidence) || confidence < MIN_FRAME_CONFIDENCE) {
        resetVoting();
        postureName.textContent = "Analyzing...";
        predictionMessage.textContent = "Confidence below threshold.";
        confidenceValue.textContent = `${confidence.toFixed(1)}%`;
        confidenceBar.style.width = `${Math.min(100, confidence)}%`;
        frameStatusText.textContent = "UNCERTAIN";
        return;
    }

    // Fast-path for high confidence frames (instant confirmation)
    const cfg = SPEED_CONFIGS[activeSpeedMode];
    voteHistory.push({ posture, confidence });
    if (voteHistory.length > cfg.voteWindow) {
        voteHistory.shift();
    }

    confidenceValue.textContent = `${confidence.toFixed(1)}%`;
    confidenceBar.style.width = `${Math.min(100, confidence)}%`;
    voteWindowProgress.textContent = `Votes: ${voteHistory.length}/${cfg.voteWindow}`;
    predictionMessage.textContent = `Stable posture verified (${confidence.toFixed(0)}% confidence).`;
    frameStatusText.textContent = "ANALYZING";

    if (sessionState === "running" && currentIntervalPosture === posture) {
        currentIntervalConfidences.push(confidence);
    }

    confirmStablePosture();
}

// ==================================================
// 10. HIGH-EFFICIENCY WEBCAM STREAM & COMPRESSION
// ==================================================

async function processFrame() {
    if (!cameraRunning || processingFrame) return;

    processingFrame = true;
    const t0 = performance.now();

    try {
        if (video.readyState < 2) return;

        const cfg = SPEED_CONFIGS[activeSpeedMode];
        const vw = video.videoWidth || 640;
        const vh = video.videoHeight || 480;

        // Downscale capture canvas for blazing <2ms JPEG encoding and fast network transfer
        const scale = Math.min(1.0, cfg.maxWidth / vw);
        const captureW = Math.round(vw * scale);
        const captureH = Math.round(vh * scale);

        if (canvas.width !== captureW || canvas.height !== captureH) {
            canvas.width = captureW;
            canvas.height = captureH;
        }

        // Keep skeleton overlay matching the high-res video
        if (skeletonOverlay.width !== vw || skeletonOverlay.height !== vh) {
            skeletonOverlay.width = vw;
            skeletonOverlay.height = vh;
        }

        captureCtx.drawImage(video, 0, 0, captureW, captureH);

        const blob = await new Promise(resolve => {
            canvas.toBlob(resolve, "image/jpeg", cfg.jpegQuality);
        });

        if (!blob) throw new Error("Frame grab failed");

        const formData = new FormData();
        formData.append("frame", blob, "stream.jpg");

        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();
        handlePrediction(data);

        // Compute roundtrip latency
        const latencyMs = Math.round(performance.now() - t0);
        rollingLatency = Math.round(rollingLatency * 0.7 + latencyMs * 0.3);
        liveLatency.textContent = `⚡ ${rollingLatency} ms`;
        if (rollingLatency < 75) {
            liveLatency.style.color = "var(--emerald-500)";
        } else if (rollingLatency < 140) {
            liveLatency.style.color = "var(--amber-500)";
        } else {
            liveLatency.style.color = "var(--rose-500)";
        }

        // Update FPS
        frameCount++;
        const now = performance.now();
        if (now - lastFrameTime >= 1000) {
            currentFps = Math.round((frameCount * 1000) / (now - lastFrameTime));
            liveFps.textContent = `${currentFps} FPS`;
            frameCount = 0;
            lastFrameTime = now;
        }

    } catch (err) {
        console.error("Frame processing error:", err);
        frameStatusText.textContent = "ERROR";
        predictionMessage.textContent = "Failed to communicate with prediction service.";
    } finally {
        processingFrame = false;
        if (cameraRunning) {
            const cfg = SPEED_CONFIGS[activeSpeedMode];
            window.setTimeout(processFrame, cfg.interval);
        }
    }
}

async function startCamera() {
    if (cameraRunning) return;

    unlockAudioEngine();

    try {
        cameraMessage.textContent = "Requesting webcam access...";

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            },
            audio: false
        });

        video.srcObject = cameraStream;
        await video.play();

        cameraRunning = true;
        resetVoting();
        confirmedPosture = null;
        lastAnnouncedPosture = null;

        cameraPlaceholder.classList.add("hidden");
        cameraStatusBadge.textContent = "LIVE";
        cameraStatusBadge.classList.add("active");
        frameStatusText.textContent = "ANALYZING";
        cameraMessage.textContent = "Optical tracking online. Maintain clear view.";

        startBtn.disabled = true;
        stopBtn.disabled = false;
        if (startBtnInPlaceholder) startBtnInPlaceholder.disabled = true;

        processFrame();
    } catch (err) {
        console.error("Webcam error:", err);
        cameraMessage.textContent = `Camera error: ${err.message || "Access denied."}`;
        showToast("Camera Error", "Could not start camera. Ensure permissions are granted.", "danger");
    }
}

function stopCamera() {
    cameraRunning = false;
    processingFrame = false;

    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    video.srcObject = null;
    cameraPlaceholder.classList.remove("hidden");
    cameraStatusBadge.textContent = "OFFLINE";
    cameraStatusBadge.classList.remove("active");
    frameStatusText.textContent = "READY";
    liveFps.textContent = "0 FPS";
    liveLatency.textContent = "⚡ 0 ms";
    hudVisibility.textContent = "VISIBILITY: --";

    startBtn.disabled = false;
    stopBtn.disabled = true;
    if (startBtnInPlaceholder) startBtnInPlaceholder.disabled = false;

    resetVoting();
    confirmedPosture = null;
    lastAnnouncedPosture = null;

    postureName.textContent = "Camera Inactive";
    postureGlyph.innerHTML = POSTURE_GLYPHS.None;
    predictionMessage.textContent = "Launch camera to analyze posture.";
    confidenceValue.textContent = "0.0%";
    confidenceBar.style.width = "0%";

    skeletonCtx.clearRect(0, 0, skeletonOverlay.width, skeletonOverlay.height);

    if (speechSupported) {
        window.speechSynthesis.cancel();
    }
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
if (startBtnInPlaceholder) startBtnInPlaceholder.addEventListener("click", startCamera);

// ==================================================
// 11. ACCURATE POSTURE DURATION TRACKER & TIMELINE
// ==================================================

function onPostureTransition(newPosture, avgConfidence) {
    const nowIso = new Date().toISOString();

    // Finalize previous posture interval
    if (currentIntervalPosture && currentIntervalDuration >= 1.0) {
        const avgConf = currentIntervalConfidences.length > 0
            ? currentIntervalConfidences.reduce((a, b) => a + b, 0) / currentIntervalConfidences.length
            : avgConfidence;

        // Record event to SQLite
        if (activeSessionId) {
            saveEventToDb(activeSessionId, currentIntervalPosture, currentIntervalStartTime, nowIso, currentIntervalDuration, avgConf);
        }

        // Add to live timeline UI
        addTimelineEntry(currentIntervalPosture, currentIntervalStartTime, nowIso, currentIntervalDuration, avgConf);
        totalTransitionsCount++;
        dashTransitions.textContent = totalTransitionsCount;
        timelineCount.textContent = `${totalTransitionsCount} Transitions`;

        // Update streak records
        if (currentIntervalPosture === "Sitting" && currentIntervalDuration > longestSittingSeconds) {
            longestSittingSeconds = currentIntervalDuration;
            dashLongestSit.textContent = formatMinSec(longestSittingSeconds);
        } else if (currentIntervalPosture === "Bending" && currentIntervalDuration > longestBendingSeconds) {
            longestBendingSeconds = currentIntervalDuration;
            dashLongestBend.textContent = formatMinSec(longestBendingSeconds);
        }
    }

    // Reset reminder flags for new posture
    sittingAlertTriggered = false;
    bendingAlertTriggered = false;

    // Start new interval
    currentIntervalPosture = newPosture;
    currentIntervalStartTime = nowIso;
    currentIntervalDuration = 0.0;
    currentIntervalConfidences = [avgConfidence];

    trackerActivePosture.textContent = newPosture;
    dashCurrentPosture.textContent = newPosture;
}

function addTimelineEntry(posture, startTime, endTime, durationSec, confidence) {
    if (timelineEmptyState) {
        timelineEmptyState.style.display = "none";
    }

    const item = document.createElement("div");
    item.className = "timeline-item";

    const pClass = posture.toLowerCase();
    const stFormatted = formatTimeString(startTime);
    const etFormatted = formatTimeString(endTime);
    const durFormatted = formatMinSec(durationSec);

    item.innerHTML = `
        <div class="timeline-left">
            <div class="timeline-badge ${pClass}">
                ${POSTURE_GLYPHS[posture] || ""}
            </div>
            <div>
                <div class="timeline-posture">${escapeHtml(posture)}</div>
                <div class="timeline-time-range">${stFormatted} &rarr; ${etFormatted}</div>
            </div>
        </div>
        <div class="text-right">
            <div class="timeline-duration font-mono">${durFormatted}</div>
            <span class="text-muted font-mono" style="font-size: 0.72rem">${confidence.toFixed(1)}% conf</span>
        </div>
    `;

    timelineList.prepend(item);
}

async function saveEventToDb(sessionId, posture, startTime, endTime, durationSec, avgConfidence) {
    try {
        await fetch("/api/session/event", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                sessionId: sessionId,
                posture: posture,
                startTime: startTime,
                endTime: endTime,
                durationSeconds: durationSec,
                avgConfidence: avgConfidence
            })
        });
    } catch (err) {
        console.error("Failed to save event to DB:", err);
    }
}

// Main Time Tracking Heartbeat Tick
function runSessionHeartbeat() {
    const now = performance.now();

    if (sessionState === "running" && lastTickTime) {
        const deltaSeconds = (now - lastTickTime) / 1000.0;

        // ONLY count time when a reliable posture is confirmed and person detected!
        if (confirmedPosture && cameraRunning) {
            totalDurationSeconds += deltaSeconds;
            currentIntervalDuration += deltaSeconds;

            if (confirmedPosture === "Standing") standingSeconds += deltaSeconds;
            else if (confirmedPosture === "Sitting") sittingSeconds += deltaSeconds;
            else if (confirmedPosture === "Squatting") squattingSeconds += deltaSeconds;
            else if (confirmedPosture === "Bending") bendingSeconds += deltaSeconds;

            checkPostureReminders(confirmedPosture, currentIntervalDuration);
        }

        updateLiveTimerDisplays();
    }

    lastTickTime = now;
    requestAnimationFrame(runSessionHeartbeat);
}

function updateLiveTimerDisplays() {
    const totalHms = formatHms(totalDurationSeconds);
    trackerSessionClock.textContent = totalHms;
    dashTotalTime.textContent = totalHms;

    const curMinSec = formatMinSec(currentIntervalDuration);
    trackerCurrentPostureDuration.textContent = curMinSec;
    dashPostureTimer.textContent = `Elapsed: ${curMinSec}`;

    trackStandClock.textContent = formatHms(standingSeconds);
    trackSitClock.textContent = formatHms(sittingSeconds);
    trackSquatClock.textContent = formatHms(squattingSeconds);
    trackBendClock.textContent = formatHms(bendingSeconds);

    dashStandingTime.textContent = formatHms(standingSeconds);
    dashSittingTime.textContent = formatHms(sittingSeconds);
    dashSquattingTime.textContent = formatHms(squattingSeconds);
    dashBendingTime.textContent = formatHms(bendingSeconds);

    if (totalDurationSeconds > 0) {
        const standPct = (standingSeconds / totalDurationSeconds) * 100;
        const sitPct = (sittingSeconds / totalDurationSeconds) * 100;
        const squatPct = (squattingSeconds / totalDurationSeconds) * 100;
        const bendPct = (bendingSeconds / totalDurationSeconds) * 100;

        dashStandingBar.style.width = `${standPct.toFixed(1)}%`;
        dashSittingBar.style.width = `${sitPct.toFixed(1)}%`;
        dashSquattingBar.style.width = `${squatPct.toFixed(1)}%`;
        dashBendingBar.style.width = `${bendPct.toFixed(1)}%`;

        dashStandingPct.textContent = `${standPct.toFixed(1)}% of total`;
        dashSittingPct.textContent = `${sitPct.toFixed(1)}% of total`;
        dashSquattingPct.textContent = `${squatPct.toFixed(1)}% of total`;
        dashBendingPct.textContent = `${bendPct.toFixed(1)}% of total`;
    }
}

// Start Session
async function startSession() {
    if (sessionState === "running") return;
    unlockAudioEngine();

    try {
        const res = await fetch("/api/session/start", { method: "POST" });
        const data = await res.json();
        if (!data.success) throw new Error(data.error);

        activeSessionId = data.session.id;
        sessionState = "running";
        sessionStartTime = new Date();
        lastTickTime = performance.now();

        if (!cameraRunning) {
            startCamera();
        }

        if (confirmedPosture) {
            currentIntervalPosture = confirmedPosture;
            currentIntervalStartTime = new Date().toISOString();
            currentIntervalDuration = 0.0;
            currentIntervalConfidences = [];
        }

        updateSessionControlsUI();
        showToast("Session Started", "Real-time posture duration tracking is active.", "success");
    } catch (err) {
        console.error("Failed to start session:", err);
        showToast("Error", "Could not create database session.", "danger");
    }
}

// Pause Session
async function pauseSession() {
    if (sessionState !== "running") return;

    try {
        await fetch("/api/session/pause", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sessionId: activeSessionId })
        });

        sessionState = "paused";
        updateSessionControlsUI();
        showToast("Session Paused", "Duration counters paused.", "warning");
    } catch (err) {
        console.error("Failed to pause session:", err);
    }
}

// Resume Session
async function resumeSession() {
    if (sessionState !== "paused") return;
    unlockAudioEngine();

    try {
        await fetch("/api/session/resume", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sessionId: activeSessionId })
        });

        sessionState = "running";
        lastTickTime = performance.now();
        updateSessionControlsUI();
        showToast("Session Resumed", "Duration tracking active.", "success");
    } catch (err) {
        console.error("Failed to resume session:", err);
    }
}

// Stop Session
async function stopSession() {
    if (sessionState === "idle") return;

    try {
        if (currentIntervalPosture && currentIntervalDuration >= 1.0) {
            const nowIso = new Date().toISOString();
            const avgConf = currentIntervalConfidences.length > 0
                ? currentIntervalConfidences.reduce((a, b) => a + b, 0) / currentIntervalConfidences.length
                : 85.0;

            await saveEventToDb(activeSessionId, currentIntervalPosture, currentIntervalStartTime, nowIso, currentIntervalDuration, avgConf);
            addTimelineEntry(currentIntervalPosture, currentIntervalStartTime, nowIso, currentIntervalDuration, avgConf);
            totalTransitionsCount++;
        }

        const response = await fetch("/api/session/stop", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                sessionId: activeSessionId,
                totalDurationSeconds: totalDurationSeconds,
                standingSeconds: standingSeconds,
                sittingSeconds: sittingSeconds,
                squattingSeconds: squattingSeconds,
                bendingSeconds: bendingSeconds,
                totalTransitions: totalTransitionsCount,
                longestSittingSeconds: longestSittingSeconds,
                longestBendingSeconds: longestBendingSeconds
            })
        });

        const resData = await response.json();
        if (resData.success) {
            showToast("Session Saved", "Monitoring session summary stored in SQLite database.", "success");
        }

        sessionState = "idle";
        activeSessionId = null;
        currentIntervalPosture = null;
        currentIntervalDuration = 0.0;

        updateSessionControlsUI();
        loadAnalyticsData();
        loadHistoryTable();
    } catch (err) {
        console.error("Failed to stop session:", err);
        showToast("Error", "Could not complete session.", "danger");
    }
}

function updateSessionControlsUI() {
    if (sessionState === "running") {
        sessionStartBtn.disabled = true;
        sessionPauseBtn.disabled = false;
        sessionPauseBtn.style.display = "inline-flex";
        sessionResumeBtn.style.display = "none";
        sessionStopBtn.disabled = false;

        trackerStatusChip.textContent = "MONITORING ACTIVE";
        trackerStatusChip.className = "status-chip chip-active";
        dashSessionState.textContent = "Monitoring active";

        navSessionDot.className = "session-indicator-dot active";
        navSessionText.textContent = "SESSION RUNNING";

        quickSessionBtnText.textContent = "Active Session";
        quickStartSessionBtn.classList.add("btn-success");
        quickStartSessionBtn.classList.remove("btn-secondary");
    } else if (sessionState === "paused") {
        sessionStartBtn.disabled = true;
        sessionPauseBtn.style.display = "none";
        sessionResumeBtn.style.display = "inline-flex";
        sessionResumeBtn.disabled = false;
        sessionStopBtn.disabled = false;

        trackerStatusChip.textContent = "PAUSED";
        trackerStatusChip.className = "status-chip chip-paused";
        dashSessionState.textContent = "Session paused";

        navSessionDot.className = "session-indicator-dot paused";
        navSessionText.textContent = "SESSION PAUSED";
    } else {
        sessionStartBtn.disabled = false;
        sessionPauseBtn.disabled = true;
        sessionPauseBtn.style.display = "inline-flex";
        sessionResumeBtn.style.display = "none";
        sessionStopBtn.disabled = true;

        trackerStatusChip.textContent = "NOT STARTED";
        trackerStatusChip.className = "status-chip chip-inactive";
        dashSessionState.textContent = "Session idle";

        navSessionDot.className = "session-indicator-dot";
        navSessionText.textContent = "NO ACTIVE SESSION";

        quickSessionBtnText.textContent = "Start Session";
        quickStartSessionBtn.classList.remove("btn-success");
        quickStartSessionBtn.classList.add("btn-secondary");
    }
}

sessionStartBtn.addEventListener("click", startSession);
sessionPauseBtn.addEventListener("click", pauseSession);
sessionResumeBtn.addEventListener("click", resumeSession);
sessionStopBtn.addEventListener("click", stopSession);

if (quickStartSessionBtn) {
    quickStartSessionBtn.addEventListener("click", () => {
        if (sessionState === "idle") {
            startSession();
        } else if (sessionState === "running") {
            switchTab("tracker");
        }
    });
}

// ==================================================
// 12. SMART ERGONOMIC POSTURE REMINDERS
// ==================================================

function checkPostureReminders(posture, intervalSeconds) {
    if (userSettings.reminder_notifications_enabled !== 1) return;

    const sitLimitSeconds = (userSettings.sitting_reminder_minutes || 30) * 60;
    const bendLimitSeconds = (userSettings.bending_reminder_minutes || 10) * 60;

    // Continuous Sitting Alert
    if (posture === "Sitting" && intervalSeconds >= sitLimitSeconds && !sittingAlertTriggered) {
        sittingAlertTriggered = true;
        const msg = `You have been sitting continuously for ${userSettings.sitting_reminder_minutes} minutes. Take a standing or walking break.`;

        showToast("Ergonomic Sitting Reminder", msg, "warning");
        triggerBrowserNotification("PoseCoach AI: Sitting Alert", msg);

        if (userSettings.reminder_voice_enabled) {
            speakPhrase(`Reminder: You have been sitting for ${userSettings.sitting_reminder_minutes} minutes. Please stand up and stretch.`, true);
        }
    }

    // Continuous Bending Alert
    if (posture === "Bending" && intervalSeconds >= bendLimitSeconds && !bendingAlertTriggered) {
        bendingAlertTriggered = true;
        const msg = `You have been bending forward for ${userSettings.bending_reminder_minutes} minutes. Adjust your posture to protect your back.`;

        showToast("Ergonomic Bending Alert", msg, "warning");
        triggerBrowserNotification("PoseCoach AI: Bending Warning", msg);

        if (userSettings.reminder_voice_enabled) {
            speakPhrase(`Warning: You have been bending for ${userSettings.bending_reminder_minutes} minutes. Please straighten your spine.`, true);
        }
    }
}

// ==================================================
// 13. USER SETTINGS FORM & SYNC
// ==================================================

async function loadUserSettings() {
    try {
        const res = await fetch("/api/settings");
        const data = await res.json();
        if (data.success && data.settings) {
            userSettings = Object.assign(userSettings, data.settings);

            document.getElementById("sitReminderInput").value = userSettings.sitting_reminder_minutes || 30;
            document.getElementById("bendReminderInput").value = userSettings.bending_reminder_minutes || 10;
            document.getElementById("browserNotifToggle").checked = userSettings.reminder_notifications_enabled === 1;
            document.getElementById("voiceReminderToggle").checked = userSettings.reminder_voice_enabled === 1;

            dashSittingThreshold.textContent = `${userSettings.sitting_reminder_minutes} min`;
            dashBendingThreshold.textContent = `${userSettings.bending_reminder_minutes} min`;
        }
    } catch (err) {
        console.error("Failed to load user settings:", err);
    }
}

async function saveUserSettings(e) {
    if (e) e.preventDefault();

    const sitVal = parseInt(document.getElementById("sitReminderInput").value, 10) || 30;
    const bendVal = parseInt(document.getElementById("bendReminderInput").value, 10) || 10;
    const notifVal = document.getElementById("browserNotifToggle").checked ? 1 : 0;
    const voiceRemVal = document.getElementById("voiceReminderToggle").checked ? 1 : 0;

    userSettings.sitting_reminder_minutes = sitVal;
    userSettings.bending_reminder_minutes = bendVal;
    userSettings.reminder_notifications_enabled = notifVal;
    userSettings.reminder_voice_enabled = voiceRemVal;

    dashSittingThreshold.textContent = `${sitVal} min`;
    dashBendingThreshold.textContent = `${bendVal} min`;

    try {
        const res = await fetch("/api/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(userSettings)
        });
        const data = await res.json();
        if (data.success) {
            showToast("Settings Saved", "Ergonomic thresholds updated.", "success");
        }
    } catch (err) {
        console.error("Settings save error:", err);
        showToast("Error", "Could not save settings.", "danger");
    }
}

// ==================================================
// 14. REAL ANALYTICS & CHART.JS INTEGRATION
// ==================================================

function updateDashboardCharts() {
    const dashCanvas = document.getElementById("dashDistributionChart");
    const dashEmpty = document.getElementById("dashChartEmpty");
    if (!dashCanvas) return;

    const dataVals = [standingSeconds, sittingSeconds, squattingSeconds, bendingSeconds];
    const hasData = dataVals.some(v => v > 0);

    if (!hasData) {
        if (dashEmpty) dashEmpty.classList.remove("hidden");
        return;
    }

    if (dashEmpty) dashEmpty.classList.add("hidden");

    if (dashChartInstance) {
        dashChartInstance.data.datasets[0].data = dataVals;
        dashChartInstance.update();
        return;
    }

    dashChartInstance = new Chart(dashCanvas, {
        type: "doughnut",
        data: {
            labels: ["Standing", "Sitting", "Squatting", "Bending"],
            datasets: [{
                data: dataVals,
                backgroundColor: ["#38bdf8", "#818cf8", "#f59e0b", "#ec4899"],
                borderWidth: 2,
                borderColor: "#101b33"
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { color: "#94a3b8", font: { family: "'DM Sans', sans-serif" } }
                }
            }
        }
    });
}

async function loadAnalyticsData() {
    try {
        const overviewRes = await fetch("/api/analytics/overview");
        const overviewData = await overviewRes.json();

        const trendsRes = await fetch("/api/analytics/trends?days=7");
        const trendsData = await trendsRes.json();

        if (overviewData.success && overviewData.summary) {
            renderAnalyticsOverview(overviewData.summary);
        }

        if (trendsData.success && trendsData.trends) {
            renderTrendsChart(trendsData.trends);
        }
    } catch (err) {
        console.error("Failed to load analytics data:", err);
    }
}

function renderAnalyticsOverview(summary) {
    const totalSessions = summary.total_sessions || 0;
    const totalSec = summary.total_monitoring_seconds || 0;
    const totalHours = (totalSec / 3600).toFixed(1);
    const totalMins = Math.round(totalSec / 60);
    const avgSec = Math.round(summary.avg_event_duration_seconds || 0);
    const maxSitMins = Math.round((summary.max_sitting_streak_seconds || 0) / 60);

    document.getElementById("statTotalSessions").textContent = totalSessions;
    document.getElementById("statTotalHours").textContent = `${totalHours}h`;
    document.getElementById("statTotalMinutes").textContent = `${totalMins} min total`;
    document.getElementById("statAvgEventDur").textContent = `${avgSec}s`;
    document.getElementById("statMaxSitStreak").textContent = `${maxSitMins}m`;

    const standSec = summary.total_standing_seconds || 0;
    const sitSec = summary.total_sitting_seconds || 0;
    const squatSec = summary.total_squatting_seconds || 0;
    const bendSec = summary.total_bending_seconds || 0;

    const hasData = (standSec + sitSec + squatSec + bendSec) > 0;

    // Pie Chart
    const pieCanvas = document.getElementById("analyticsPieChart");
    const pieEmpty = document.getElementById("analyticsPieEmpty");

    if (!hasData) {
        if (pieEmpty) pieEmpty.classList.remove("hidden");
    } else {
        if (pieEmpty) pieEmpty.classList.add("hidden");
        const pieData = [standSec, sitSec, squatSec, bendSec];

        if (analyticsPieInstance) {
            analyticsPieInstance.data.datasets[0].data = pieData;
            analyticsPieInstance.update();
        } else {
            analyticsPieInstance = new Chart(pieCanvas, {
                type: "doughnut",
                data: {
                    labels: ["Standing", "Sitting", "Squatting", "Bending"],
                    datasets: [{
                        data: pieData,
                        backgroundColor: ["#38bdf8", "#818cf8", "#f59e0b", "#ec4899"],
                        borderWidth: 2,
                        borderColor: "#101b33"
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: "bottom", labels: { color: "#94a3b8" } }
                    }
                }
            });
        }
    }

    // Bar Chart
    const barCanvas = document.getElementById("analyticsBarChart");
    const barEmpty = document.getElementById("analyticsBarEmpty");

    if (!hasData) {
        if (barEmpty) barEmpty.classList.remove("hidden");
    } else {
        if (barEmpty) barEmpty.classList.add("hidden");
        const barDataMins = [
            (standSec / 60).toFixed(1),
            (sitSec / 60).toFixed(1),
            (squatSec / 60).toFixed(1),
            (bendSec / 60).toFixed(1)
        ];

        if (analyticsBarInstance) {
            analyticsBarInstance.data.datasets[0].data = barDataMins;
            analyticsBarInstance.update();
        } else {
            analyticsBarInstance = new Chart(barCanvas, {
                type: "bar",
                data: {
                    labels: ["Standing", "Sitting", "Squatting", "Bending"],
                    datasets: [{
                        label: "Total Duration (Minutes)",
                        data: barDataMins,
                        backgroundColor: ["#38bdf8", "#818cf8", "#f59e0b", "#ec4899"],
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: "rgba(255, 255, 255, 0.05)" },
                            ticks: { color: "#94a3b8" }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: "#94a3b8" }
                        }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
        }
    }
}

function renderTrendsChart(trends) {
    const trendCanvas = document.getElementById("analyticsTrendChart");
    const trendEmpty = document.getElementById("analyticsTrendEmpty");

    if (!trends || trends.length === 0) {
        if (trendEmpty) trendEmpty.classList.remove("hidden");
        return;
    }

    if (trendEmpty) trendEmpty.classList.add("hidden");

    const labels = trends.map(t => t.session_date);
    const sitHours = trends.map(t => ((t.sitting_seconds || 0) / 3600).toFixed(2));
    const standHours = trends.map(t => ((t.standing_seconds || 0) / 3600).toFixed(2));
    const squatHours = trends.map(t => ((t.squatting_seconds || 0) / 3600).toFixed(2));
    const bendHours = trends.map(t => ((t.bending_seconds || 0) / 3600).toFixed(2));

    if (analyticsTrendInstance) {
        analyticsTrendInstance.data.labels = labels;
        analyticsTrendInstance.data.datasets[0].data = sitHours;
        analyticsTrendInstance.data.datasets[1].data = standHours;
        analyticsTrendInstance.data.datasets[2].data = squatHours;
        analyticsTrendInstance.data.datasets[3].data = bendHours;
        analyticsTrendInstance.update();
    } else {
        analyticsTrendInstance = new Chart(trendCanvas, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    { label: "Sitting (h)", data: sitHours, backgroundColor: "#818cf8", borderRadius: 4 },
                    { label: "Standing (h)", data: standHours, backgroundColor: "#38bdf8", borderRadius: 4 },
                    { label: "Squatting (h)", data: squatHours, backgroundColor: "#f59e0b", borderRadius: 4 },
                    { label: "Bending (h)", data: bendHours, backgroundColor: "#ec4899", borderRadius: 4 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        stacked: true,
                        grid: { display: false },
                        ticks: { color: "#94a3b8" }
                    },
                    y: {
                        stacked: true,
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#94a3b8" }
                    }
                },
                plugins: {
                    legend: {
                        position: "top",
                        labels: { color: "#94a3b8", font: { family: "'DM Sans', sans-serif" } }
                    }
                }
            }
        });
    }
}

// ==================================================
// 15. HISTORY & SESSION REPORTS TABLE
// ==================================================

async function loadHistoryTable() {
    const tbody = document.getElementById("historyTableBody");
    if (!tbody) return;

    tbody.innerHTML = '<tr><td colspan="9" class="text-center py-4">Fetching recorded sessions...</td></tr>';

    try {
        const res = await fetch("/api/analytics/sessions?limit=50");
        const data = await res.json();

        if (!data.success || !data.sessions || data.sessions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="9" class="text-center py-4 text-muted">No completed sessions found in the database. Complete your first monitoring session!</td></tr>';
            return;
        }

        tbody.innerHTML = "";

        data.sessions.forEach(s => {
            const tr = document.createElement("tr");
            const totFormatted = formatHms(s.total_duration_seconds);
            const standFormatted = formatMinSec(s.standing_seconds);
            const sitFormatted = formatMinSec(s.sitting_seconds);
            const squatFormatted = formatMinSec(s.squatting_seconds);
            const bendFormatted = formatMinSec(s.bending_seconds);

            let dateStr = s.start_time;
            try {
                dateStr = new Date(s.start_time).toLocaleString();
            } catch {}

            tr.innerHTML = `
                <td class="font-mono">#${s.id}</td>
                <td>${escapeHtml(dateStr)}</td>
                <td class="font-mono font-bold">${totFormatted}</td>
                <td class="font-mono text-accent">${standFormatted}</td>
                <td class="font-mono text-warning">${sitFormatted}</td>
                <td class="font-mono">${squatFormatted}</td>
                <td class="font-mono text-danger">${bendFormatted}</td>
                <td class="font-mono text-center">${s.total_transitions}</td>
                <td>
                    <div class="table-actions">
                        <a href="/api/reports/session/${s.id}/pdf" class="btn btn-primary btn-sm" download>
                            PDF
                        </a>
                        <a href="/api/reports/session/${s.id}/csv" class="btn btn-outline btn-sm" download>
                            CSV
                        </a>
                    </div>
                </td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.error("Failed to load history table:", err);
        tbody.innerHTML = '<tr><td colspan="9" class="text-center py-4 text-danger">Failed to retrieve session history.</td></tr>';
    }
}

// ==================================================
// 16. INITIALIZATION & CLEANUP
// ==================================================

window.addEventListener("DOMContentLoaded", () => {
    loadUserSettings();
    loadAnalyticsData();
    requestAnimationFrame(runSessionHeartbeat);

    // Warm-up audio engine on any initial click
    document.addEventListener("click", unlockAudioEngine, { once: true });
    console.log("PoseCoach AI Turbo Engine initialized.");
});

window.addEventListener("pagehide", () => {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
    }
    if (speechSupported) {
        window.speechSynthesis.cancel();
    }
});
