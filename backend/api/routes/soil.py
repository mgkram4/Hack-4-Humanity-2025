from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from services.soil_service import SoilService

router = APIRouter()

@router.get("/soil")
async def get_soil_data(city: str):
    try:
        soil_service = SoilService()
        data = soil_service.get_soil_data(city)
        return JSONResponse(
            status_code=200,
            content=data
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@router.get("/", response_class=HTMLResponse)
async def home():
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>NASA Soil Data Analysis</title>
            <style>
                * { box-sizing: border-box; margin: 0; padding: 0; }
                body {
                    font-family: system-ui, -apple-system, sans-serif;
                    background: #f5f5f5;
                    padding: 20px;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .header {
                    margin-bottom: 20px;
                }
                .header h1 {
                    font-size: 2.5rem;
                    color: #1a1a1a;
                }
                .header p {
                    color: #666;
                    font-size: 1.2rem;
                }
                .input-group {
                    display: flex;
                    gap: 10px;
                    margin-bottom: 20px;
                }
                .input-group input {
                    flex: 1;
                    padding: 12px;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    font-size: 1rem;
                }
                .input-group button {
                    padding: 12px 24px;
                    background: #4361ee;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 1rem;
                    transition: background 0.3s;
                }
                .input-group button:hover { background: #3851e0; }
                .input-group button:disabled {
                    background: #ccc;
                    cursor: not-allowed;
                }
                .error {
                    background: #fee2e2;
                    border: 1px solid #ef4444;
                    color: #b91c1c;
                    padding: 12px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    display: none;
                }
                .metrics {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }
                .metric-card {
                    background: white;
                    padding: 20px;
                    border-radius: 12px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .metric-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 10px;
                }
                .metric-value {
                    font-size: 2rem;
                    font-weight: bold;
                    color: #1a1a1a;
                }
                .metric-label {
                    color: #666;
                    font-size: 0.9rem;
                }
                .loading {
                    display: none;
                    text-align: center;
                    padding: 20px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>NASA Soil Data Analysis</h1>
                    <p>Real-time soil monitoring using NASA POWER data</p>
                </div>

                <div class="input-group">
                    <input type="text" id="cityInput" placeholder="Enter city name">
                    <button id="fetchButton">Get Soil Data</button>
                </div>

                <div class="error" id="errorMessage"></div>
                <div class="loading" id="loadingIndicator">Loading...</div>

                <div class="metrics" id="metricsContainer">
                    <div class="metric-card">
                        <div class="metric-header">
                            <span class="metric-label">Surface Soil Moisture</span>
                            <span class="metric-value" id="surfaceMoistureValue">--</span>
                        </div>
                        <div class="metric-label">Percentage (%)</div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-header">
                            <span class="metric-label">Root Zone Moisture</span>
                            <span class="metric-value" id="rootMoistureValue">--</span>
                        </div>
                        <div class="metric-label">Percentage (%)</div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-header">
                            <span class="metric-label">Profile Moisture</span>
                            <span class="metric-value" id="deepMoistureValue">--</span>
                        </div>
                        <div class="metric-label">Percentage (%)</div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-header">
                            <span class="metric-label">Surface Temperature</span>
                            <span class="metric-value" id="surfaceTempValue">--</span>
                        </div>
                        <div class="metric-label">Celsius (°C)</div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-header">
                            <span class="metric-label">Subsurface Temperature</span>
                            <span class="metric-value" id="subsurfaceTempValue">--</span>
                        </div>
                        <div class="metric-label">Celsius (°C)</div>
                    </div>
                </div>
            </div>

            <script>
                const cityInput = document.getElementById('cityInput');
                const fetchButton = document.getElementById('fetchButton');
                const errorMessage = document.getElementById('errorMessage');
                const loadingIndicator = document.getElementById('loadingIndicator');
                
                const surfaceMoistureValue = document.getElementById('surfaceMoistureValue');
                const rootMoistureValue = document.getElementById('rootMoistureValue');
                const deepMoistureValue = document.getElementById('deepMoistureValue');
                const surfaceTempValue = document.getElementById('surfaceTempValue');
                const subsurfaceTempValue = document.getElementById('subsurfaceTempValue');

                function showError(message) {
                    errorMessage.textContent = message;
                    errorMessage.style.display = 'block';
                }

                function hideError() {
                    errorMessage.style.display = 'none';
                }

                function setLoading(isLoading) {
                    fetchButton.disabled = isLoading;
                    loadingIndicator.style.display = isLoading ? 'block' : 'none';
                    cityInput.disabled = isLoading;
                }

                function updateMetrics(data) {
                    surfaceMoistureValue.textContent = `${data.surface_moisture}%`;
                    rootMoistureValue.textContent = `${data.root_moisture}%`;
                    deepMoistureValue.textContent = `${data.deep_moisture}%`;
                    surfaceTempValue.textContent = `${data.surface_temp}°C`;
                    subsurfaceTempValue.textContent = `${data.subsurface_temp}°C`;
                }

                async function fetchSoilData() {
                    const city = cityInput.value.trim();
                    if (!city) {
                        showError('Please enter a city name');
                        return;
                    }

                    hideError();
                    setLoading(true);

                    try {
                        const response = await fetch(`/api/v1/soil?city=${encodeURIComponent(city)}`);
                        if (!response.ok) {
                            const error = await response.json();
                            throw new Error(error.error || 'Failed to fetch soil data');
                        }
                        const data = await response.json();
                        updateMetrics(data);
                    } catch (err) {
                        showError(err.message);
                    } finally {
                        setLoading(false);
                    }
                }

                fetchButton.addEventListener('click', fetchSoilData);
                cityInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        fetchSoilData();
                    }
                });
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)