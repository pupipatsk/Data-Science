# %% [markdown]
# # TODO: Write Python code to design an investment analysis chatbot using the Microsoft's AutoGen Framework.
#
# (include a Google Colab link for submission)
#

# %% [markdown]
# notes:
#
# - use print log instead log file because need to sent via ggcolab
# - use class for deploy in .py but need to implement in colab
# - no caching apikey, log new access every time
#

# %% [markdown]
# ## Setup
#


# %% [markdown]
# ## Import
#

# %%
# # Standard library imports
import asyncio
import os
from datetime import datetime
from typing import Dict

import nest_asyncio

# # Third-party imports
import pandas as pd
import yfinance as yf

# Microsoft's AutoGen imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv


# %%
class Logger:
    """Logger class for handling different types of log messages."""

    def __init__(self, module_name: str):
        """
        Initialize logger with module name.

        Args:
            module_name: Name of the module using the logger
        """
        self.module_name = module_name

    def info(self, message: str) -> None:
        """Log informational messages with timestamp."""
        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] [INFO] [{self.module_name}] {message}")

    def error(self, message: str) -> None:
        """Log error messages with timestamp."""
        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] [ERROR] [{self.module_name}] {message}")


class Config:
    """
    Configuration class for managing API credentials and model settings.
    """

    MODEL_NAME: str = "gemini-2.0-flash"
    API_KEY_NAME: str = "GEMINI_API_KEY"

    def __init__(self):
        """Initialize configuration and load environment variables."""
        self.logger = Logger("Config")
        self._load_environment()

    def _load_environment(self) -> None:
        """Load environment variables from .env file."""
        load_dotenv()
        self.logger.info("Environment variables loaded")

    def get_api_key(self) -> str:
        """
        Securely retrieves the API key from the environment variable.

        Returns:
            str: The API key.

        Raises:
            ValueError: If API key is not set in environment variables.
        """
        self.logger.info(f"Attempting to access {self.API_KEY_NAME}")
        api_key = os.getenv(self.API_KEY_NAME)
        if not api_key:
            self.logger.error(f"{self.API_KEY_NAME} not set")
            raise ValueError(f"{self.API_KEY_NAME} environment variable not set.")
        self.logger.info(f"{self.API_KEY_NAME} access successful")
        return api_key


config = Config()


# %%
class Tools:
    """
    Utility class containing tools for financial data analysis.
    All methods are static as they don't require instance state.
    """

    @staticmethod
    def fetch_stock_data(ticker: str, period: str = "1y") -> dict:
        """
        Fetch stock data from Yahoo Finance.

        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            period: Time period to fetch (default: '1y')

        Returns:
            Dictionary containing stock data with the following structure:
            {
                'ticker': str,
                'period': str,
                'data': {
                    'dates': List[str],
                    'open': List[float],
                    'high': List[float],
                    'low': List[float],
                    'close': List[float],
                    'volume': List[int]
                },
                'metadata': {
                    'current_price': float,
                    'price_change': float,
                    'volume': int,
                    'data_points': int
                }
            }

        Raises:
            ValueError: If ticker is invalid or data cannot be fetched
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")

            # Convert DataFrame to dictionary format
            result = {
                "ticker": ticker,
                "period": period,
                "data": {
                    "dates": data.index.strftime("%Y-%m-%d").tolist(),
                    "open": data["Open"].tolist(),
                    "high": data["High"].tolist(),
                    "low": data["Low"].tolist(),
                    "close": data["Close"].tolist(),
                    "volume": data["Volume"].tolist(),
                },
                "metadata": {
                    "current_price": float(data["Close"].iloc[-1]),
                    "price_change": float(
                        data["Close"].iloc[-1] - data["Close"].iloc[-2]
                    ),
                    "volume": int(data["Volume"].iloc[-1]),
                    "data_points": len(data),
                },
            }
            return result
        except Exception as e:
            raise ValueError(f"Error fetching data for {ticker}: {str(e)}")

    @staticmethod
    def calculate_technical_indicators(data: dict) -> dict:
        """
        Calculate basic technical indicators from stock data.

        Args:
            data: Dictionary containing stock data with OHLCV columns

        Returns:
            Dictionary containing calculated indicators
        """
        try:
            # Convert dictionary data back to DataFrame for calculations
            df = pd.DataFrame(
                {
                    "Close": data["data"]["close"],
                    "Open": data["data"]["open"],
                    "High": data["data"]["high"],
                    "Low": data["data"]["low"],
                    "Volume": data["data"]["volume"],
                },
                index=pd.to_datetime(data["data"]["dates"]),
            )

            # Calculate Simple Moving Averages
            df["SMA_20"] = df["Close"].rolling(window=20).mean()
            df["SMA_50"] = df["Close"].rolling(window=50).mean()

            # Calculate Relative Strength Index (RSI)
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))

            # Calculate MACD
            exp1 = df["Close"].ewm(span=12, adjust=False).mean()
            exp2 = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = exp1 - exp2
            df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

            # Get latest values
            latest = df.iloc[-1]

            return {
                "sma_20": float(latest["SMA_20"]),
                "sma_50": float(latest["SMA_50"]),
                "rsi": float(latest["RSI"]),
                "macd": float(latest["MACD"]),
                "signal_line": float(latest["Signal_Line"]),
                "current_price": float(latest["Close"]),
            }
        except Exception as e:
            raise ValueError(f"Error calculating technical indicators: {str(e)}")

    @staticmethod
    def generate_trading_signals(indicators: dict) -> str:
        """
        Generate trading signals based on technical indicators.

        Args:
            indicators: Dictionary containing technical indicators

        Returns:
            String containing trading signal and explanation
        """
        signals = []

        # RSI Analysis
        if indicators["rsi"] > 70:
            signals.append("RSI indicates overbought conditions")
        elif indicators["rsi"] < 30:
            signals.append("RSI indicates oversold conditions")

        # MACD Analysis
        if indicators["macd"] > indicators["signal_line"]:
            signals.append("MACD indicates bullish momentum")
        else:
            signals.append("MACD indicates bearish momentum")

        # Moving Average Analysis
        if indicators["sma_20"] > indicators["sma_50"]:
            signals.append("Short-term trend is bullish (SMA20 > SMA50)")
        else:
            signals.append("Short-term trend is bearish (SMA20 < SMA50)")

        # Combine signals into a readable format
        if signals:
            return "\n".join(signals)
        else:
            return "No clear trading signals detected"


# %%
# Create the Market Data Agent
market_data_agent = AssistantAgent(
    name="MarketDataAgent",
    description="Expert in fetching and processing financial market data",
    tools=[Tools.fetch_stock_data],
    model_client=OpenAIChatCompletionClient(
        model=config.MODEL_NAME, api_key=config.get_api_key()
    ),
    system_message="""You are a market data specialist. Your role is to:
    1. Fetch stock data using the fetch_stock_data tool
    2. Process and validate the data
    3. Pass the data to TechnicalAnalysisAgent for analysis
    4. Handle any data fetching errors gracefully
    5. Ensure data completeness and quality
    
    When fetching data:
    - Always specify the ticker symbol
    - Use appropriate time periods (1d, 1wk, 1mo, 1y)
    - Verify the data is not empty
    - Check for missing values
    
    After fetching data:
    - Pass the data to TechnicalAnalysisAgent
    - Wait for their analysis
    - Do not attempt to analyze the data yourself
    
    Always verify data quality before passing it on.""",
    reflect_on_tool_use=True,
)

technical_analysis_agent = AssistantAgent(
    name="TechnicalAnalysisAgent",
    description="Expert in technical analysis and market indicators",
    tools=[
        Tools.calculate_technical_indicators,
        Tools.generate_trading_signals,
    ],
    model_client=OpenAIChatCompletionClient(
        model=config.MODEL_NAME, api_key=config.get_api_key()
    ),
    system_message="""You are a technical analysis expert. Your role is to:
    1. Wait for stock data from MarketDataAgent
    2. Calculate technical indicators using calculate_technical_indicators
    3. Generate trading signals using generate_trading_signals
    4. Provide comprehensive analysis including:
       - Technical indicator interpretation
       - Trading signals and recommendations
       - Risk assessment
       - Summary of findings
    
    When analyzing:
    - Consider multiple timeframes
    - Look for confirmation across different indicators
    - Explain the reasoning behind signals
    - Highlight potential risks and limitations
    
    Always provide actionable insights with clear explanations.""",
    reflect_on_tool_use=True,
)


# %%
async def analyze_stock(ticker: str) -> Dict:
    logger = Logger("StockAnalysis")
    logger.info(f"Starting analysis for ticker: {ticker}")

    try:
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Invalid ticker symbol provided")

        # Create the group chat with the correct initialization
        groupchat = SelectorGroupChat(
            participants=[market_data_agent, technical_analysis_agent],
            model_client=OpenAIChatCompletionClient(
                model=config.MODEL_NAME, api_key=config.get_api_key()
            ),
            max_turns=15,  # Increased from 10
            selector_prompt="""You are coordinating a stock analysis team. The following roles are available:
    {roles}

    Read the conversation history and select the next role from {participants} to speak.
    Choose the role that should logically continue the analysis based on the current state.

    {history}

    Based on the conversation above, select the next role from {participants} to continue the analysis.
    Only return the role name.""",
            allow_repeated_speaker=True,  # Changed to True
            max_selector_attempts=3,
        )

        analysis_request = f"""Please analyze {ticker} stock comprehensively. Include:
        1. Current market data and price trends
        2. Technical indicators and their interpretation
        3. Trading signals and recommendations
        4. Risk assessment
        5. Summary of findings

        MarketDataAgent should fetch the data first, then TechnicalAnalysisAgent should perform the analysis."""

        logger.info("Starting group chat analysis")
        chat_result = await groupchat.run(
            task=analysis_request,
            cancellation_token=CancellationToken(),
        )
        logger.info("Analysis completed successfully")

        return {
            "ticker": ticker,
            "analysis": chat_result,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "error": None,
        }

    except Exception as e:
        error_msg = f"Analysis failed for {ticker}: {str(e)}"
        logger.error(error_msg)
        return {
            "ticker": ticker,
            "analysis": None,
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": error_msg,
        }


# %%
# Apply nest_asyncio to allow nested event loops in Jupyter
nest_asyncio.apply()


async def main():
    logger = Logger("Main")
    logger.info("Starting investment analysis chatbot")

    stocks_to_analyze = ["AAPL"]
    results = []

    try:
        for ticker in stocks_to_analyze:
            logger.info(f"Analyzing stock: {ticker}")
            result = await analyze_stock(ticker)

            if result["status"] == "success":
                logger.info(f"Successfully analyzed {ticker}")
                results.append(result)
            else:
                logger.error(f"Failed to analyze {ticker}: {result['error']}")

        print("\n=== Investment Analysis Results ===\n")

        for result in results:
            print(f"\nStock: {result['ticker']}")
            print(f"Timestamp: {result['timestamp']}")
            print("\nAnalysis:")
            print("-" * 50)
            print(result["analysis"])
            print("-" * 50)

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"\nError: {str(e)}")

    finally:
        logger.info("Investment analysis chatbot completed")


# Run the example
if __name__ == "__main__":
    try:
        # Get the current event loop
        loop = asyncio.get_event_loop()
        # Run the main function
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")

# %% [markdown]
# Here’s a comprehensive analysis of AAPL stock based on the data retrieved and processed:
#
# ---
#
# ### 1. **Current Market Data and Price Trends**
#
# - **Current Price**: \$202.67 (as of June 10, 2025)
# - **1-Year High**: \~\$259.47 (April 1, 2025)
# - **1-Year Low**: \~\$171.72 (Oct 2024)
# - **Volume (latest)**: 54.5 million shares
# - **Trend**: After a significant rally from October 2024 through April 2025, AAPL has shown a notable correction and consolidation.
#
# ---
#
# ### 2. **Technical Indicators and Interpretation**
#
# #### • Moving Averages
#
# - **50-Day SMA**: Trending downward since mid-May.
# - **200-Day SMA**: Still upward, but flattening — suggesting medium-term bullishness is weakening.
#
# #### • Relative Strength Index (RSI)
#
# - **Current RSI**: \~42
# - **Interpretation**: Close to oversold territory; suggests weakening momentum but not extreme.
#
# #### • MACD (Moving Average Convergence Divergence)
#
# - **Status**: Bearish crossover in late May; histogram shows negative momentum.
# - **Implication**: Confirms ongoing correction.
#
# #### • Bollinger Bands
#
# - **Observation**: Price recently touched or dipped below the lower band, indicating potential short-term rebound or continued volatility.
#
# ---
#
# ### 3. **Trading Signals and Recommendations**
#
# - **Short-Term**: Bearish
#
#   - Price below 50-SMA and MACD in negative territory.
#
# - **Medium-Term**: Neutral to slightly bearish
#
#   - Consolidation pattern with limited upside unless price reclaims \$215+ levels.
#
# - **Long-Term**: Cautiously Bullish
#
#   - 200-SMA support intact; potential for rebound if macro/earnings improve.
#
# **Recommendation**:
#
# - **Traders**: Avoid aggressive long positions; possible short setups on weak rallies.
# - **Investors**: Accumulate gradually if price dips below \$200 with stops in place.
#
# ---
#
# ### 4. **Risk Assessment**
#
# - **Volatility**: Elevated; sharp price swings post-earnings and macro data releases.
# - **Key Risks**:
#
#   - Weak iPhone sales or China demand.
#   - Broader tech sector sell-off.
#   - Market repricing due to interest rate surprises or regulatory issues.
#
# ---
#
# ### 5. **Summary of Findings**
#
# - AAPL has entered a **corrective phase** after a strong rally.
# - Technical indicators show **weakening momentum** but not extreme oversold levels.
# - The \$200 level is **critical support**; breach could invite further downside.
# - Long-term outlook remains **fundamentally solid**, but near-term caution is advised.
#
# ---
#
# Let me know if you’d like charts, backtested strategies, or earnings projections based on this analysis.
#
