# Standard library imports
import asyncio
import os
from datetime import datetime
from typing import Dict, List, Literal

import nest_asyncio

# Third-party imports
import pandas as pd
import yfinance as yf

# Microsoft's AutoGen imports
from autogen_core import (
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    message_handler,
    type_subscription,
)
from autogen_core.models import (
    ChatCompletionClient,
    ModelInfo,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# -- Data Models --


class Price(BaseModel):
    """
    Represents a single OHLCV (Open, High, Low, Close, Volume) price data point.

    Attributes:
        time (str): Timestamp of the price data point.
        open (float): Opening price.
        high (float): Highest price during the period.
        low (float): Lowest price during the period.
        close (float): Closing price.
        volume (int): Trading volume for the period.
    """

    time: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class StockData(BaseModel):
    """Model for basic stock market data"""

    ticker: str
    period: str
    prices: List[Price] = Field(
        ...,
        description="List of OHLCV price data points",
    )


class TechnicalIndicators(BaseModel):
    """Model for technical analysis indicators"""

    ticker: str
    sma_20: float = Field(..., description="20-day Simple Moving Average")
    sma_50: float = Field(..., description="50-day Simple Moving Average")
    rsi: float = Field(..., description="Relative Strength Index")
    macd: float = Field(..., description="Moving Average Convergence Divergence")
    macd_signal_line: float = Field(..., description="MACD Signal Line")


class TradingSignals(BaseModel):
    """Model for trading recommendations"""

    ticker: str
    signals: List[str] = Field(..., description="List of trading signals")
    recommendation: Literal["buy", "sell", "hold"] = Field(
        ..., description="Trading recommendation"
    )
    confidence: float = Field(
        ..., description="Confidence level of the recommendation (0-1)"
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        ..., description="Risk level of the trade"
    )


# -- Settings --


class Logger:
    """Logger class for handling different types of log messages."""

    LOG_FILE = "log.log"

    def __init__(self, module_name: str):
        """
        Initialize logger with module name.

        Args:
            module_name: Name of the module using the logger
        """
        self.module_name = module_name

    def _write_log(self, level: str, message: str) -> None:
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}][{level}][{self.module_name}] {message}"
        print(log_entry)
        try:
            with open(self.LOG_FILE, "a", encoding="utf-8") as f:
                f.write(log_entry + "\n")
        except Exception as e:
            print(f"[{timestamp}][ERROR][Logger] Failed to write to log file: {e}")

    def info(self, message: str) -> None:
        """Log informational messages with timestamp."""
        self._write_log("INFO", message)

    def warning(self, message: str) -> None:
        """Log warning messages with timestamp."""
        self._write_log("WARNING", message)

    def error(self, message: str) -> None:
        """Log error messages with timestamp."""
        self._write_log("ERROR", message)


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


# -- Tools --


class StockDataTools:
    """
    Tools for fetching and processing stock market data.
    """

    def __init__(self):
        self.logger = Logger(self.__class__.__name__)

    def validate_inputs(self, ticker: str, period: str | None = None) -> None:
        """Validate ticker symbol and optional period."""
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Invalid ticker symbol provided")

        if period is not None:
            valid_periods = [
                "1d",
                "5d",
                "1mo",
                "3mo",
                "6mo",
                "1y",
                "2y",
                "5y",
                "10y",
                "ytd",
                "max",
            ]
            if period not in valid_periods:
                raise ValueError(
                    f"Invalid period. Must be one of: {', '.join(valid_periods)}"
                )

    def get_data_from_yfinance(self, ticker: str, period: str) -> pd.DataFrame:
        """
        Fetch raw stock data from Yahoo Finance.

        Args:
            ticker (str): The stock ticker symbol
            period (str): The time period for data

        Returns:
            pd.DataFrame: Raw stock data from Yahoo Finance
        """
        self.logger.info(
            f"External API request: service=YahooFinance, action=fetch_data, ticker={ticker}, period={period}"
        )

        # Validate inputs
        self.validate_inputs(ticker, period)

        try:
            # Initialize Ticker object
            stock = yf.Ticker(ticker)

            # Fetch historical data
            try:
                data = stock.history(period=period)
            except Exception as e:
                self.logger.error(f"Error fetching price data for {ticker}: {str(e)}")
                raise ValueError(f"Error fetching price data for {ticker}: {str(e)}")

            # Verify data is not empty
            if data.empty:
                self.logger.error(f"No price data available for ticker: {ticker}")
                raise ValueError(f"No price data available for ticker: {ticker}")

            # Convert column names to lowercase
            data.columns = [col.lower() for col in data.columns]

            self.logger.info(
                f"External API response: service=YahooFinance, action=fetch_data, ticker={ticker}, status=success, rows={len(data)}"
            )
            return data  # pd.DataFrame

        except Exception as e:
            error_msg = f"Unexpected error fetching data for {ticker}: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def fetch_stock_data(self, ticker: str, period: str) -> str:
        """
        Fetch stock data from Yahoo Finance.

        Args:
            ticker (str): The stock ticker symbol
            period (str): The time period for data

        Returns:
            str: JSON string containing stock data and metadata

        Raises:
            ValueError: If ticker is invalid or no data is found
        """
        self.logger.info(f"Fetching stock data [ticker={ticker}] [period={period}]")

        # Validate inputs
        self.validate_inputs(ticker, period)

        try:
            data = self.get_data_from_yfinance(ticker, period)  # pd.DataFrame

            stock_data = StockData(
                ticker=ticker,
                period=period,
                prices=[
                    Price(
                        time=str(index.date()),
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=int(row["volume"]),
                    )
                    for index, row in data.iterrows()
                ],
            )

            self.logger.info(
                f"Data processing complete: ticker={ticker}, period={period}, data_points={len(data)}"
            )
            return stock_data.model_dump_json(indent=2)

        except ValueError:
            # Re-raise ValueError as is since we've already logged it
            raise
        except Exception as e:
            error_msg = f"Error processing stock data for {ticker}: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)


class TechnicalAnalysisTools:
    """
    Tools for calculating technical indicators and generating trading signals.
    """

    def __init__(self):
        self.logger = Logger(self.__class__.__name__)

    def calculate_technical_indicators(self, data_json: str) -> str:
        """
        Calculate technical indicators from stock data.

        Args:
            data_json (str): JSON string containing stock data

        Returns:
            str: JSON string containing technical indicators

        Raises:
            ValueError: If data is invalid or calculations fail
        """
        self.logger.info("Calculating technical indicators")

        def _validate_input(data: StockData):
            self.logger.info("Validating input data for required fields and length")
            if not data.prices:
                self.logger.error("Price data is empty.")
                raise ValueError("Price data is empty.")
            if len(data.prices) < 50:
                self.logger.error("Insufficient data points for technical analysis")
                raise ValueError("Insufficient data points for technical analysis")

        def _create_dataframe(data: StockData) -> pd.DataFrame:
            self.logger.info("Creating DataFrame from input data")
            df = pd.DataFrame([p.model_dump() for p in data.prices])
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            return df

        def _calculate_sma(df):
            self.logger.info("Calculating SMA20 & SMA50")
            df["sma_20"] = df["close"].rolling(window=20).mean()
            df["sma_50"] = df["close"].rolling(window=50).mean()

        def _calculate_rsi(df):
            self.logger.info("Calculating RSI")
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))

        def _calculate_macd(df):
            self.logger.info("Calculating MACD")
            exp1 = df["close"].ewm(span=12, adjust=False).mean()
            exp2 = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"] = exp1 - exp2
            df["macd_signal_line"] = df["macd"].ewm(span=9, adjust=False).mean()

        def _extract_latest_indicators(
            df: pd.DataFrame, ticker: str
        ) -> TechnicalIndicators:
            self.logger.info("Extracting latest indicator values")
            latest = df.iloc[-1]
            return TechnicalIndicators(
                ticker=ticker,
                sma_20=float(latest["sma_20"]),
                sma_50=float(latest["sma_50"]),
                rsi=float(latest["rsi"]),
                macd=float(latest["macd"]),
                macd_signal_line=float(latest["macd_signal_line"]),
            )

        try:
            # Parse JSON input into StockData
            data = StockData.model_validate_json(data_json)
            _validate_input(data)
            df = _create_dataframe(data)
            _calculate_sma(df)
            _calculate_rsi(df)
            _calculate_macd(df)
            result = _extract_latest_indicators(df, data.ticker)
            self.logger.info(f"Successfully calculated indicators for {data.ticker}")
            return result.model_dump_json(indent=2)

        except KeyError as e:
            self.logger.error(f"Missing data field: {str(e)}")
            raise ValueError(f"Missing required data field: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            raise ValueError(f"Error calculating technical indicators: {str(e)}")

    def generate_trading_signals(self, indicators_json: str) -> str:
        """
        Generate trading signals based on technical indicators.

        Args:
            indicators_json (str): JSON string containing technical indicators

        Returns:
            str: JSON string containing trading signals and recommendations

        Raises:
            ValueError: If indicators are invalid or signal generation fails
        """
        self.logger.info("Generating trading signals")

        def analyze_rsi(rsi):
            self.logger.info(f"Analyzing RSI: {rsi}")
            if rsi > 70:
                return "RSI indicates overbought conditions"
            elif rsi < 30:
                return "RSI indicates oversold conditions"
            return None

        def analyze_macd(macd, macd_signal_line):
            self.logger.info(
                f"Analyzing MACD: {macd} vs Signal Line: {macd_signal_line}"
            )
            if macd > macd_signal_line:
                return "MACD indicates bullish momentum"
            else:
                return "MACD indicates bearish momentum"

        def analyze_sma(sma_20, sma_50):
            self.logger.info(f"Analyzing SMAs: 20-day {sma_20} vs 50-day {sma_50}")
            if sma_20 > sma_50:
                return "Short-term trend is bullish (SMA20 > SMA50)"
            else:
                return "Short-term trend is bearish (SMA20 < SMA50)"

        try:
            # Parse JSON input into TechnicalIndicators
            indicators = TechnicalIndicators.model_validate_json(indicators_json)
            signals = []

            rsi_signal = analyze_rsi(indicators.rsi)
            if rsi_signal:
                signals.append(rsi_signal)

            macd_signal = analyze_macd(indicators.macd, indicators.macd_signal_line)
            if macd_signal:
                signals.append(macd_signal)

            sma_signal = analyze_sma(indicators.sma_20, indicators.sma_50)
            if sma_signal:
                signals.append(sma_signal)

            signal_text = (
                "\n".join(signals) if signals else "No clear trading signals detected"
            )
            self.logger.info(f"Generated trading signals: {signal_text}")

            # Determine recommendation based on signals
            recommendation = "hold"
            if any("bullish" in s.lower() for s in signals):
                recommendation = "buy"
            elif any("bearish" in s.lower() for s in signals):
                recommendation = "sell"

            # Calculate confidence based on number of confirming signals
            confidence = 0.8 if len(signals) > 2 else 0.5

            # Determine risk level
            risk_level = "medium"
            if any(
                "overbought" in s.lower() or "oversold" in s.lower() for s in signals
            ):
                risk_level = "high"

            result = TradingSignals(
                ticker=indicators.ticker,
                signals=signals,
                recommendation=recommendation,
                confidence=confidence,
                risk_level=risk_level,
            )
            return result.model_dump_json(indent=2)

        except Exception as e:
            self.logger.error(f"Error generating trading signals: {str(e)}")
            raise ValueError(f"Error generating trading signals: {str(e)}")


class Tools:
    """
    Main interface for financial analysis tools.
    Delegates to specialized tool classes while maintaining the same external interface.
    """

    def __init__(self):
        self.stock_data_tools = StockDataTools()
        self.technical_analysis_tools = TechnicalAnalysisTools()

        # Create function tools with strict=True
        self.fetch_stock_data = FunctionTool(
            self.stock_data_tools.fetch_stock_data,
            description="Fetch stock data from Yahoo Finance",
            strict=True,
        )

        self.calculate_technical_indicators = FunctionTool(
            self.technical_analysis_tools.calculate_technical_indicators,
            description="Calculate technical indicators from stock data",
            strict=True,
        )

        self.generate_trading_signals = FunctionTool(
            self.technical_analysis_tools.generate_trading_signals,
            description="Generate trading signals based on technical indicators",
            strict=True,
        )

    def get_tools(self) -> List[FunctionTool]:
        """Get all function tools."""
        return [
            self.fetch_stock_data,
            self.calculate_technical_indicators,
            self.generate_trading_signals,
        ]


# -- Message Protocol --


class StockDataMessage(BaseModel):
    """Message containing stock data"""

    content: StockData


class TechnicalIndicatorsMessage(BaseModel):
    """Message containing technical indicators"""

    content: TechnicalIndicators


class TradingSignalsMessage(BaseModel):
    """Message containing trading signals"""

    content: TradingSignals


# -- Topics --
market_data_topic_type = "MarketDataAgent"
technical_indicators_topic_type = "TechnicalIndicatorsAgent"
trading_signals_topic_type = "TradingSignalsAgent"
user_topic_type = "UserAgent"

# -- Agent Definitions --


@type_subscription(topic_type=market_data_topic_type)
class MarketDataAgent(RoutedAgent):
    """Agent responsible for fetching stock data"""

    def __init__(self, model_client: ChatCompletionClient, tools: Tools) -> None:
        super().__init__("Market Data Agent that fetches stock data")
        self.tools = tools
        self.logger = Logger(self.__class__.__name__)
        self._model_client = model_client
        self._system_message = SystemMessage(
            content=(
                "You are a market data analyst. Given a stock ticker, fetch and analyze the stock data.\n"
                "Use the provided tools to fetch the data and process it.\n"
                "Ensure the data is valid and complete before proceeding."
            )
        )

    @message_handler
    async def handle_analysis_request(self, message: str, ctx: MessageContext) -> None:
        """Handle analysis request and fetch stock data"""
        try:
            self.logger.info(f"Received analysis request: {message}")

            # Use model client to process the request
            prompt = f"Please analyze the following stock: {message}"
            llm_result = await self._model_client.create(
                messages=[
                    self._system_message,
                    UserMessage(content=prompt, source=self.id.key),
                ],
                cancellation_token=ctx.cancellation_token,
            )
            response = llm_result.content
            assert isinstance(response, str)

            # Extract ticker from message (assuming format like "Analyze AAPL")
            ticker = message.split()[-1].upper()

            # Fetch stock data using tools
            stock_data_json = self.tools.stock_data_tools.fetch_stock_data(ticker, "1y")

            self.logger.info(f"Successfully fetched data for {ticker}")
            print(f"{'-' * 80}\n{self.id.type}:\nFetched stock data for {ticker}")

            # Publish to next agent
            await self.publish_message(
                StockDataMessage(
                    content=StockData.model_validate_json(stock_data_json)
                ),
                topic_id=TopicId(technical_indicators_topic_type, source=self.id.key),
            )

        except Exception as e:
            self.logger.error(f"Error in MarketDataAgent: {str(e)}")
            print(f"Error fetching data: {str(e)}")


@type_subscription(topic_type=technical_indicators_topic_type)
class TechnicalIndicatorsAgent(RoutedAgent):
    """Agent responsible for calculating technical indicators"""

    def __init__(self, model_client: ChatCompletionClient, tools: Tools) -> None:
        super().__init__("Technical Indicators Agent that calculates indicators")
        self.tools = tools
        self.logger = Logger(self.__class__.__name__)
        self._model_client = model_client
        self._system_message = SystemMessage(
            content=(
                "You are a technical analysis expert. Given stock data, calculate and analyze technical indicators.\n"
                "Use the provided tools to calculate indicators and provide insights.\n"
                "Focus on key indicators like SMA, RSI, and MACD."
            )
        )

    @message_handler
    async def handle_stock_data(
        self, message: StockDataMessage, ctx: MessageContext
    ) -> None:
        """Handle stock data and calculate technical indicators"""
        try:
            self.logger.info("Calculating technical indicators from stock data")

            # Use model client to analyze the data
            prompt = (
                f"Please analyze the technical indicators for {message.content.ticker}"
            )
            llm_result = await self._model_client.create(
                messages=[
                    self._system_message,
                    UserMessage(content=prompt, source=self.id.key),
                ],
                cancellation_token=ctx.cancellation_token,
            )
            response = llm_result.content
            assert isinstance(response, str)

            # Calculate indicators using tools
            indicators_json = (
                self.tools.technical_analysis_tools.calculate_technical_indicators(
                    message.content.model_dump_json()
                )
            )

            self.logger.info("Successfully calculated technical indicators")
            print(f"{'-' * 80}\n{self.id.type}:\nCalculated technical indicators")

            # Publish to next agent
            await self.publish_message(
                TechnicalIndicatorsMessage(
                    content=TechnicalIndicators.model_validate_json(indicators_json)
                ),
                topic_id=TopicId(trading_signals_topic_type, source=self.id.key),
            )

        except Exception as e:
            self.logger.error(f"Error in TechnicalIndicatorsAgent: {str(e)}")
            print(f"Error calculating indicators: {str(e)}")


@type_subscription(topic_type=trading_signals_topic_type)
class TradingSignalsAgent(RoutedAgent):
    """Agent responsible for generating trading signals"""

    def __init__(self, model_client: ChatCompletionClient, tools: Tools) -> None:
        super().__init__("Trading Signals Agent that generates trading signals")
        self.tools = tools
        self.logger = Logger(self.__class__.__name__)
        self._model_client = model_client
        self._system_message = SystemMessage(
            content=(
                "You are a trading signals expert. Given technical indicators, generate trading signals and recommendations.\n"
                "Use the provided tools to analyze indicators and generate signals.\n"
                "Provide clear recommendations with confidence levels and risk assessment."
            )
        )

    @message_handler
    async def handle_indicators(
        self, message: TechnicalIndicatorsMessage, ctx: MessageContext
    ) -> None:
        """Handle technical indicators and generate trading signals"""
        try:
            self.logger.info("Generating trading signals from technical indicators")

            # Use model client to analyze the indicators
            prompt = f"Please analyze the trading signals for {message.content.ticker}"
            llm_result = await self._model_client.create(
                messages=[
                    self._system_message,
                    UserMessage(content=prompt, source=self.id.key),
                ],
                cancellation_token=ctx.cancellation_token,
            )
            response = llm_result.content
            assert isinstance(response, str)

            # Generate signals using tools
            signals_json = self.tools.technical_analysis_tools.generate_trading_signals(
                message.content.model_dump_json()
            )

            self.logger.info("Successfully generated trading signals")
            print(f"{'-' * 80}\n{self.id.type}:\nGenerated trading signals")

            # Publish to user agent
            await self.publish_message(
                TradingSignalsMessage(
                    content=TradingSignals.model_validate_json(signals_json)
                ),
                topic_id=TopicId(user_topic_type, source=self.id.key),
            )

        except Exception as e:
            self.logger.error(f"Error in TradingSignalsAgent: {str(e)}")
            print(f"Error generating signals: {str(e)}")


@type_subscription(topic_type=user_topic_type)
class UserAgent(RoutedAgent):
    """Agent that outputs the final analysis to the user"""

    def __init__(self) -> None:
        super().__init__("User Agent that displays final results")
        self.logger = Logger(self.__class__.__name__)

    @message_handler
    async def handle_final_analysis(
        self, message: TradingSignalsMessage, ctx: MessageContext
    ) -> None:
        """Handle final trading signals and display to user"""
        try:
            self.logger.info("Received final analysis results")

            # Access the Pydantic model directly
            signals_data = message.content

            print(f"\n{'-' * 80}")
            print(f"{self.id.type} received final analysis:")
            print(f"Ticker: {signals_data.ticker}")
            print(f"Recommendation: {signals_data.recommendation}")
            print(f"Confidence: {signals_data.confidence}")
            print(f"Risk Level: {signals_data.risk_level}")
            print(f"Signals: {signals_data.signals}")
            print(f"{'-' * 80}")

        except Exception as e:
            self.logger.error(f"Error in UserAgent: {str(e)}")
            print(f"Error displaying results: {str(e)}")


async def analyze_stock(
    ticker: str, tools: Tools, model_client: ChatCompletionClient
) -> Dict:
    """Run the sequential workflow for stock analysis"""
    logger = Logger("analyze_stock")
    logger.info(f"Starting sequential workflow analysis for ticker: {ticker}")

    # Create runtime
    logger.info("Creating SingleThreadedAgentRuntime instance")
    runtime = SingleThreadedAgentRuntime()

    # Register agents
    logger.info("Registering MarketDataAgent")
    await MarketDataAgent.register(
        runtime,
        type=market_data_topic_type,
        factory=lambda: MarketDataAgent(model_client=model_client, tools=tools),
    )

    logger.info("Registering TechnicalIndicatorsAgent")
    await TechnicalIndicatorsAgent.register(
        runtime,
        type=technical_indicators_topic_type,
        factory=lambda: TechnicalIndicatorsAgent(
            model_client=model_client, tools=tools
        ),
    )

    logger.info("Registering TradingSignalsAgent")
    await TradingSignalsAgent.register(
        runtime,
        type=trading_signals_topic_type,
        factory=lambda: TradingSignalsAgent(model_client=model_client, tools=tools),
    )

    logger.info("Registering UserAgent")
    await UserAgent.register(runtime, type=user_topic_type, factory=lambda: UserAgent())

    # Start runtime
    logger.info("Starting agent runtime")
    runtime.start()

    try:
        # Publish initial message to start the workflow
        logger.info(
            f"Publishing initial message to start workflow for ticker: {ticker}"
        )
        await runtime.publish_message(
            f"Analyze stock data for {ticker}",
            topic_id=TopicId(market_data_topic_type, source="default"),
        )

        # Wait for workflow to complete
        logger.info("Waiting for workflow to complete (runtime.stop_when_idle)")
        await runtime.stop_when_idle()

        logger.info(f"Sequential workflow completed for {ticker}")
        return {"status": "success", "message": f"Analysis completed for {ticker}"}

    except Exception as e:
        logger.error(f"Error in sequential workflow: {str(e)}")
        return {"status": "error", "error": str(e)}


class MarketDataResponse(BaseModel):
    """Structured output model for MarketDataAgent"""

    thoughts: str = Field(..., description="Analysis thoughts and reasoning")
    data: StockData = Field(..., description="Stock market data")


class TechnicalAnalysisResponse(BaseModel):
    """Structured output model for TechnicalAnalysisAgent"""

    thoughts: str = Field(..., description="Analysis thoughts and reasoning")
    indicators: TechnicalIndicators
    signals: TradingSignals
    summary: str
    status: Literal["success", "error"]
    error_message: str | None = None


# Apply nest_asyncio to allow nested event loops in Jupyter
nest_asyncio.apply()


async def main():
    logger = Logger("Main")
    logger.info("Starting investment analysis chatbot")

    # Initialize configuration
    config = Config()
    logger.info("Configuration initialized")

    # Initialize global gemini model_client
    # ver. custom
    # model_client = OpenAIChatCompletionClient(
    #     model=config.MODEL_NAME,
    #     api_key=config.get_api_key(),
    # )
    # ver. Docs
    model_client = OpenAIChatCompletionClient(
        model="gemini-2.0-flash-lite",
        model_info=ModelInfo(
            vision=True,
            function_calling=True,
            json_output=True,
            family="unknown",
            structured_output=True,
        ),
        api_key=config.get_api_key(),
    )

    logger.info("Global gemini model_client initialized")

    # Initialize tools
    tools = Tools()
    logger.info("Tools initialized")

    stocks_to_analyze = ["AAPL"]

    try:
        for ticker in stocks_to_analyze:
            logger.info(f"Analyzing stock: {ticker}")
            result = await analyze_stock(ticker, tools, model_client)

            if result["status"] == "success":
                logger.info(f"Analysis completed for {ticker}")
                logger.info(f"Analysis result: {result['message']}")
            else:
                logger.error(f"Analysis failed for {ticker}: {result['error']}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"\nError: {str(e)}")

    finally:
        logger.info("Closing model_client")
        await model_client.close()


# Run the example
if __name__ == "__main__":
    logger = Logger("__main__")
    logger.info("Script started: __main__")
    try:
        # Get the current event loop
        loop = asyncio.get_event_loop()
        logger.info("Event loop acquired, running main()")
        # Run the main function
        loop.run_until_complete(main())
        logger.info("Main execution finished")
    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user (KeyboardInterrupt)")
        print("\nAnalysis interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\nUnexpected error: {str(e)}")
    finally:
        logger.info("Script exiting: __main__")
