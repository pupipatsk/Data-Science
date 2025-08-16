# # Standard library imports
import asyncio
import os
from datetime import datetime
from typing import Dict, List, Literal, Union

import nest_asyncio

# # Third-party imports
import pandas as pd
import yfinance as yf

# Microsoft's AutoGen imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

# Add to imports section after the existing imports
from pydantic import BaseModel
from typing_extensions import TypedDict


# Define response models
class StockData(BaseModel):
    ticker: str
    period: str
    current_price: float
    price_change: float
    volume: int
    data_points: int


class TechnicalIndicators(BaseModel):
    sma_20: float
    sma_50: float
    rsi: float
    macd: float
    signal_line: float
    current_price: float


class TradingSignals(BaseModel):
    signals: List[str]
    recommendation: Literal["buy", "sell", "hold"]
    confidence: float
    risk_level: Literal["low", "medium", "high"]


class AnalysisResponse(BaseModel):
    market_data: StockData
    technical_analysis: TechnicalIndicators
    trading_signals: TradingSignals
    summary: str


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
model_client = OpenAIChatCompletionClient(
    model=config.MODEL_NAME, api_key=config.get_api_key()
)


class StockDataDict(TypedDict):
    ticker: str
    period: str
    data: Dict[str, List[Union[str, float, int]]]
    metadata: Dict[str, Union[float, int]]


class TechnicalIndicatorsDict(TypedDict):
    sma_20: float
    sma_50: float
    rsi: float
    macd: float
    signal_line: float
    current_price: float


class Tools:
    """
    Utility class containing tools for financial data analysis.
    All methods are static as they don't require instance state.
    """

    # Add class-level logger
    logger = Logger("Tools")

    @staticmethod
    def fetch_stock_data(ticker: str, period: str = "1y") -> StockDataDict:
        """
        Fetch stock data from Yahoo Finance.

        Args:
            ticker (str): The stock ticker symbol
            period (str, optional): The time period for data. Defaults to "1y".

        Returns:
            StockDataDict: Dictionary containing stock data and metadata
        """
        Tools.logger.info(f"Fetching stock data for {ticker} over {period}")
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            if data.empty:
                Tools.logger.error(f"No data found for ticker {ticker}")
                raise ValueError(f"No data found for ticker {ticker}")

            # Convert DataFrame to dictionary format
            result: StockDataDict = {
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
            Tools.logger.info(
                f"Successfully fetched {len(data)} data points for {ticker}"
            )
            return result
        except Exception as e:
            Tools.logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise ValueError(f"Error fetching data for {ticker}: {str(e)}")

    @staticmethod
    def calculate_technical_indicators(data: StockDataDict) -> TechnicalIndicatorsDict:
        """
        Calculate basic technical indicators from stock data.

        Args:
            data (StockDataDict): Dictionary containing stock data

        Returns:
            TechnicalIndicatorsDict: Dictionary containing technical indicators
        """
        Tools.logger.info(f"Calculating technical indicators for {data['ticker']}")
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
            Tools.logger.info("Calculating SMAs")
            df["SMA_20"] = df["Close"].rolling(window=20).mean()
            df["SMA_50"] = df["Close"].rolling(window=50).mean()

            # Calculate Relative Strength Index (RSI)
            Tools.logger.info("Calculating RSI")
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))

            # Calculate MACD
            Tools.logger.info("Calculating MACD")
            exp1 = df["Close"].ewm(span=12, adjust=False).mean()
            exp2 = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = exp1 - exp2
            df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

            # Get latest values
            latest = df.iloc[-1]

            result: TechnicalIndicatorsDict = {
                "sma_20": float(latest["SMA_20"]),
                "sma_50": float(latest["SMA_50"]),
                "rsi": float(latest["RSI"]),
                "macd": float(latest["MACD"]),
                "signal_line": float(latest["Signal_Line"]),
                "current_price": float(latest["Close"]),
            }
            Tools.logger.info(
                f"Successfully calculated indicators for {data['ticker']}"
            )
            return result
        except Exception as e:
            Tools.logger.error(f"Error calculating technical indicators: {str(e)}")
            raise ValueError(f"Error calculating technical indicators: {str(e)}")

    @staticmethod
    def generate_trading_signals(indicators: TechnicalIndicatorsDict) -> str:
        """
        Generate trading signals based on technical indicators.

        Args:
            indicators (TechnicalIndicatorsDict): Dictionary containing technical indicators

        Returns:
            str: Trading signals and recommendations
        """
        Tools.logger.info("Generating trading signals")
        signals = []

        # RSI Analysis
        Tools.logger.info(f"Analyzing RSI: {indicators['rsi']}")
        if indicators["rsi"] > 70:
            signals.append("RSI indicates overbought conditions")
        elif indicators["rsi"] < 30:
            signals.append("RSI indicates oversold conditions")

        # MACD Analysis
        Tools.logger.info(
            f"Analyzing MACD: {indicators['macd']} vs Signal Line: {indicators['signal_line']}"
        )
        if indicators["macd"] > indicators["signal_line"]:
            signals.append("MACD indicates bullish momentum")
        else:
            signals.append("MACD indicates bearish momentum")

        # Moving Average Analysis
        Tools.logger.info(
            f"Analyzing SMAs: 20-day {indicators['sma_20']} vs 50-day {indicators['sma_50']}"
        )
        if indicators["sma_20"] > indicators["sma_50"]:
            signals.append("Short-term trend is bullish (SMA20 > SMA50)")
        else:
            signals.append("Short-term trend is bearish (SMA20 < SMA50)")

        # Combine signals into a readable format
        result = "\n".join(signals) if signals else "No clear trading signals detected"
        Tools.logger.info(f"Generated trading signals: {result}")
        return result


# Create the Market Data Agent
market_data_agent = AssistantAgent(
    name="MarketDataAgent",
    description="Expert in fetching and processing financial market data",
    tools=[Tools.fetch_stock_data],
    model_client=model_client,
    system_message="""You are a market data specialist. Your role is to:
    1. Fetch stock data using the fetch_stock_data tool
    2. Process and validate the data
    3. Return the data in a clear, readable format
    
    When fetching data:
    - Always specify the ticker symbol
    - Use appropriate time periods (1d, 1wk, 1mo, 1y)
    - Verify the data is not empty
    - Check for missing values
    - Provide a summary of the data fetched""",
    reflect_on_tool_use=True,
)

technical_analysis_agent = AssistantAgent(
    name="TechnicalAnalysisAgent",
    description="Expert in technical analysis and market indicators",
    tools=[
        Tools.calculate_technical_indicators,
        Tools.generate_trading_signals,
    ],
    model_client=model_client,
    system_message="""You are a technical analysis expert. Your role is to:
    1. Calculate technical indicators using calculate_technical_indicators
    2. Generate trading signals using generate_trading_signals
    3. Provide comprehensive analysis and insights
    
    When analyzing:
    - Consider multiple timeframes
    - Look for confirmation across different indicators
    - Explain the reasoning behind signals
    - Highlight potential risks and limitations
    - Provide clear, actionable recommendations""",
    reflect_on_tool_use=True,
)


async def analyze_stock(ticker: str) -> Dict:
    logger = Logger("StockAnalysis")
    logger.info(f"Starting analysis for ticker: {ticker}")

    try:
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Invalid ticker symbol provided")

        groupchat = SelectorGroupChat(
            participants=[market_data_agent, technical_analysis_agent],
            model_client=model_client,
            max_turns=15,
            selector_prompt="""You are coordinating a stock analysis team. The following roles are available:
    {roles}

    Read the conversation history and select the next role from {participants} to speak.
    Choose the role that should logically continue the analysis based on the current state.

    {history}

    Based on the conversation above, select the next role from {participants} to continue the analysis.
    Only return the role name.""",
            allow_repeated_speaker=True,
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

        # Use Console to stream the messages
        result = await Console(
            groupchat.run_stream(
                task=analysis_request,
                cancellation_token=CancellationToken(),
            ),
            output_stats=True,
        )

        logger.info("Analysis completed successfully")

        return {
            "ticker": ticker,
            "analysis": str(result.messages[-1].content)
            if result.messages
            else "No analysis completed",
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "error": None,
        }

    except Exception as e:
        error_msg = f"Analysis failed for {ticker}: {str(e)}"
        logger.error(error_msg)
        return {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": error_msg,
        }


# Apply nest_asyncio to allow nested event loops in Jupyter
nest_asyncio.apply()


async def main():
    logger = Logger("Main")
    logger.info("Starting investment analysis chatbot")

    stocks_to_analyze = ["AAPL"]

    try:
        for ticker in stocks_to_analyze:
            logger.info(f"Analyzing stock: {ticker}")
            await analyze_stock(ticker)

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"\nError: {str(e)}")

    finally:
        logger.info("Investment analysis chatbot completed")
        await model_client.close()


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
