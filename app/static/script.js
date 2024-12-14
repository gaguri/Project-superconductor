document.addEventListener('DOMContentLoaded', function () {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const loading = document.querySelector('.loading');
    const results = document.getElementById('results');

    // Drag and drop handlers
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#2ecc71';
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#3498db';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#3498db';

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file');
            return;
        }

        const reader = new FileReader();
        reader.onload = function (e) {
            // Display preview
            const img = document.createElement('img');
            img.src = e.target.result;
            dropZone.innerHTML = '';
            dropZone.appendChild(img);
            loading.style.display = 'flex';
        };
        reader.readAsDataURL(file);
        analyzePhaseDigram(file);
    }

    function analyzePhaseDigram(file) {
        // This is where you would normally make an API call to your backend
        // For demo purposes, we'll use random values
        const data = new FormData()
        data.append('file', file)


        fetch('/calculate-parameters', {
                method: 'POST',
                body: data
            }
        )
            .then(value => value.json())
            .then(
            value => {
                const result = value
                console.log(result)
                updateResults(result.params)
                loading.style.display = 'none';
                results.style.display = 'block';
            }
        )
    }

    function updateResults(parameters) {
        // Update progress bars and values with animation
        animateProgress('D', 100, 'D', parameters.D + ' D');
        animateProgress('V', 100, 'V', parameters.V + ' V');
        animateProgress('td', 100, 'td', parameters.td + ' td');
        animateProgress('tp', 100, 'tp', parameters.tp + ' tp');
    }

    function animateProgress(progressId, percentage, valueId, finalValue) {
        const progress = document.getElementById(progressId);
        const value = document.getElementById(valueId);
        let current = 0;

        const interval = setInterval(() => {
            if (current >= percentage) {
                clearInterval(interval);
                value.textContent = finalValue;
            } else {
                current += 1;
                progress.style.width = current + '%';
            }
        }, 10);
    }
});