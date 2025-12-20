// --- DOM Elements ---
const landingPage = document.getElementById('landing-page');
const dashboardSection = document.getElementById('dashboard-section');
const videoElement = document.getElementById('webcam');
const resultMood = document.getElementById('result-mood');
const resultConf = document.getElementById('result-conf');
const resultSuggestion = document.getElementById('result-suggestion');
const resultEmoji = document.getElementById('result-emoji');
const noHistoryMsg = document.getElementById('no-history-msg');
const chartCanvas = document.getElementById('moodChart');

let myChart = null;

const savedData = localStorage.getItem('moodData');
let moodHistoryData = savedData ? JSON.parse(savedData) : { 
    'Happy': 0, 'Sad': 0, 'Neutral': 0, 'Shocked': 0, 'Angry': 0 
};

const moodEmojis = { 
    "Happy": "😊", "Sad": "😔", "Neutral": "😐", 
    "Shocked": "😲", "Angry": "😡"
};

function enterDashboard() {
    landingPage.classList.add('hidden');
    dashboardSection.classList.remove('hidden');
    startCamera();
}

function goHome() {
    const stream = videoElement.srcObject;
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    videoElement.srcObject = null;
    dashboardSection.classList.add('hidden');
    landingPage.classList.remove('hidden');
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
    } catch (err) {
        alert("Camera access denied!");
        goHome();
    }
}

async function analyzeMood() {
    resultMood.innerText = "Analyzing...";
    resultMood.style.color = "#94a3b8";
    resultConf.innerText = "--";
    resultSuggestion.innerText = "Connecting to AI model...";
    
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('image', blob, 'scan.jpg');

        const roleDropdown = document.getElementById('ai-role');
        const selectedRole = roleDropdown ? roleDropdown.value : "Friend"; 
        formData.append('role', selectedRole); 

        try {
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const data = await response.json();

            if (data.error) {
                resultMood.innerText = "Error";
                resultSuggestion.innerText = data.error;
            } else {
                updateDashboard(data);
            }
        } catch (error) {
            resultMood.innerText = "Connection Failed";
            resultSuggestion.innerText = "Check Python Backend";
        }
    }, 'image/jpeg');
}

function updateDashboard(data) {
    const detectedMood = data.mood;
    resultMood.innerText = detectedMood;
    resultMood.style.color = "#22d3ee";
    resultConf.innerText = data.confidence;
    resultSuggestion.innerText = data.suggestion;
    resultEmoji.innerText = moodEmojis[detectedMood] || "🤖";
    updateGraph(detectedMood);
}

function updateGraph(mood) {
    noHistoryMsg.style.display = 'none';
    chartCanvas.style.display = 'block';

    if (mood === "Surprise") mood = "Shocked";
    if (mood === "Anger") mood = "Angry"; 
    mood = mood.charAt(0).toUpperCase() + mood.slice(1);

    if (moodHistoryData.hasOwnProperty(mood)) moodHistoryData[mood]++;
    else return;

    localStorage.setItem('moodData', JSON.stringify(moodHistoryData));

    const labels = Object.keys(moodHistoryData);
    const dataValues = Object.values(moodHistoryData);

    if (myChart) {
        myChart.data.datasets[0].data = dataValues;
        myChart.update();
    } else {
        createChart(labels, dataValues);
    }
}

function createChart(labels, dataValues) {
    const ctx = chartCanvas.getContext('2d');
    myChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                label: '# of Scans',
                data: dataValues,
                backgroundColor: ['#FCD34D', '#60A5FA', '#9CA3AF', '#818CF8', '#EF4444'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'right', labels: { color: 'white' } } }
        }
    });
}

function resetGraph() {
    for (let key in moodHistoryData) moodHistoryData[key] = 0;
    localStorage.removeItem('moodData'); // Clear Memory

    if (myChart) { myChart.destroy(); myChart = null; }
    chartCanvas.style.display = 'none';
    noHistoryMsg.style.display = 'block';
    resultMood.innerText = "Waiting...";
    resultSuggestion.innerText = "Click 'Analyze Now' to get a suggestion.";
}

window.onload = function() {
    const totalScans = Object.values(moodHistoryData).reduce((a, b) => a + b, 0);
    if (totalScans > 0) {
        noHistoryMsg.style.display = 'none';
        chartCanvas.style.display = 'block';
        createChart(Object.keys(moodHistoryData), Object.values(moodHistoryData));
    }
};