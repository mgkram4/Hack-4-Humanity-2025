// Initialize charts when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Common chart options
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            }
        },
        scales: {
            y: {
                beginAtZero: true
            }
        }
    };

    // Weather chart
    const weatherCtx = document.getElementById('weather-chart').getContext('2d');
    new Chart(weatherCtx, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            datasets: [{
                data: [30, 40, 20, 35, 25, 30],
                borderColor: '#2563eb',
                tension: 0.4
            }]
        },
        options: commonOptions
    });

    // Soil chart
    const soilCtx = document.getElementById('soil-chart').getContext('2d');
    new Chart(soilCtx, {
        type: 'bar',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            datasets: [{
                data: [65, 75, 70, 80, 85, 82],
                backgroundColor: '#22c55e'
            }]
        },
        options: commonOptions
    });

    // Price chart
    const priceCtx = document.getElementById('price-chart').getContext('2d');
    new Chart(priceCtx, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr'],
            datasets: [{
                data: [100, 120, 90, 110],
                borderColor: '#f59e0b',
                tension: 0.4
            }]
        },
        options: commonOptions
    });
});