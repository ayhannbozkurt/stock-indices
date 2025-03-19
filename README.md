# Stock Indices

This repository ontains a Streamlit application for analyzing the performance of bank stocks in the Turkish stock market in comparison with the Bank Index (XBANK). The application provides various analyses including correlation, beta coefficients, performance metrics, clustering, and structural break detection.

## Features

### Overview Dashboard
- **Banking Sector Overview**: A quick glance at the banking sector's overall performance with metrics comparing banks against the XBANK index
- **Best and Worst Performing Banks**: Highlights the top and bottom performers in the sector

### Basic Analyses
- **Normalized Price Movements**: Visualizes price trends of all banks normalized to a starting value of 100
- **Correlation with Bank Index**: Shows how closely each bank's stock movement correlates with the XBANK index
- **Beta Coefficients**: Displays each bank's beta (volatility compared to the market)
- **Annual Performance**: Compares the performance of each bank over the last year

### Advanced Analyses
- **Structural Break Analysis**: Detects significant change points in the price trends using the PELT algorithm
- **Clustering Analysis**: Groups banks with similar movement patterns using K-means clustering
- **Rolling Correlation**: Shows how the correlation between banks and the index changes over time
- **Performance Ratio**: Visualizes the performance ratio between banks and the XBANK index
- **Beta-Performance Relationship**: Explores the relationship between beta values and performance
- **Monthly Performance Heatmap**: Displays monthly performance of selected banks with a color-coded heatmap

## Technical Implementation

### Data Sources
- Stock price data is fetched using the Yahoo Finance API (yfinance)
- The application analyzes both individual bank stocks and the XBANK index

### Data Analysis Methods
- **Correlation Analysis**: Pearson correlation between daily returns
- **Beta Calculation**: Covariance-based approach to calculate market sensitivity
- **Structural Break Detection**: Implemented using the ruptures library with PELT algorithm
- **Clustering**: K-means clustering with automatic determination of optimal cluster count
- **Performance Metrics**: Calculation of various performance indicators including total return, volatility, etc.

### Visualization Techniques
- Custom color schemes for better data visualization
- Interactive components allowing users to select specific banks for comparison
- Heatmaps for correlation matrices and monthly performance
- Combination of line, bar, and scatter plots for different analysis types

## Project Structure

- `main.py`: The main Streamlit application file containing the UI components
- `data_utils.py`: Contains functions for data fetching and analysis
- `visualization.py`: Contains all visualization functions
- `requirements.txt`: Lists all required packages

## Installation

```bash
# Clone the repository
git clone https://github.com/ayhannbozkurt/stock-indices.git
cd stock-indices

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

## Usage

After launching the application with `streamlit run main.py`, you can:

1. View the banking sector overview on the main dashboard
2. Explore basic and advanced analyses through the tabbed interface
3. Select specific banks for comparison in various analyses
4. Adjust parameters like the rolling window size for correlation analysis

## Data Caching

- The application implements caching with a 6-hour time-to-live (TTL) to improve performance and limit API calls
- Analysis results are cached to ensure quick navigation between different visualizations

## Customization

You can modify the following aspects of the application:
- Time period for analysis by changing the `days` parameter in `load_data()`
- Number of structural break points by adjusting `n_bkps` parameter
- Visual aspects like colors and layout by editing the CSS in `main.py`


## Future Improvements

- Adding more technical indicators for stock analysis
- Implementing predictive models for price forecasting
- Adding fundamental analysis data to complement technical analysis
- Supporting more markets and indices for comparison
- Exporting analysis results to various formats (PDF, Excel, etc.)

## Acknowledgements

- Turkish Stock Exchange for providing market data through Yahoo Finance
- The Streamlit team for their excellent framework for data applications
