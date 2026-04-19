document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('text-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loader = document.getElementById('loader');
    const resultContainer = document.getElementById('result-container');
    const predictionBadge = document.getElementById('prediction-badge');
    const predictionProb = document.getElementById('prediction-prob');
    const confidenceFill = document.getElementById('confidence-fill');

    // Mute standard colors defined in CSS
    const HATE_COLOR = '#8b3a3a';
    const OK_COLOR = '#556b2f';

    analyzeBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (!text) return;

        // Reset UI
        analyzeBtn.disabled = true;
        loader.classList.remove('hidden');
        resultContainer.classList.add('hidden');
        confidenceFill.style.width = '0%';

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });

            const data = await response.json();

            if (response.ok) {
                displayResult(data);
            } else {
                alert(`Error: ${data.error || 'Something went wrong.'}`);
            }
        } catch (error) {
            console.error('Error calling prediction API:', error);
            alert('Failed to communicate with the server.');
        } finally {
            analyzeBtn.disabled = false;
            loader.classList.add('hidden');
        }
    });

    function displayResult(data) {
        // Expected data: { label: "HATE" | "NOT HATE", probability: 0.95 }
        
        const isHate = data.label === 'HATE';
        const probPercentage = (data.probability * 100).toFixed(1);

        // Update badge
        predictionBadge.textContent = data.label;
        predictionBadge.className = 'badge'; // reset
        predictionBadge.classList.add(isHate ? 'hate' : 'ok');

        // Update probability text
        predictionProb.textContent = `${probPercentage}%`;
        
        // Update progress bar
        confidenceFill.style.backgroundColor = isHate ? HATE_COLOR : OK_COLOR;
        
        // Reveal container and animate bar
        resultContainer.classList.remove('hidden');
        
        // Small delay to ensure CSS transition fires since element just became visible
        setTimeout(() => {
            confidenceFill.style.width = `${probPercentage}%`;
        }, 50);
    }
});
