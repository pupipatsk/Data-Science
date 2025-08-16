# Standard library imports
import asyncio
import os
from datetime import datetime
from typing import Dict, List, Literal, Union

import nest_asyncio

# Third-party imports
import pandas as pd
import yfinance as yf

# Microsoft's AutoGen imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field


# Define data models using Pydantic BaseModel for consistency
class BaseStockModel(BaseModel):
    """Base class for all stock-related data models"""

    ticker: str
    current_price: float


class StockMetadata(BaseModel):
    """Model for stock metadata information"""

    price_change: float
    volume: int
    data_points: int


class StockData(BaseStockModel):
    """Model for basic stock market data"""

    period: str
    data: Dict[str, List[Union[str, float, int]]] = Field(
        ...,
        description="Dictionary containing stock price data with keys: dates, open, high, low, close, volume",
        example={
            "dates": ["2024-01-01"],
            "open": [150.0],
            "high": [155.0],
            "low": [148.0],
            "close": [152.0],
            "volume": [1000000],
        },
    )
    metadata: StockMetadata


class TechnicalIndicators(BaseStockModel):
    """Model for technical analysis indicators"""

    sma_20: float
    sma_50: float
    rsi: float
    macd: float
    signal_line: float


class TradingSignals(BaseStockModel):
    """Model for trading recommendations"""

    signals: List[str]
    recommendation: Literal["buy", "sell", "hold"]
    confidence: float
    risk_level: Literal["low", "medium", "high"]


class AnalysisResponse(BaseStockModel):
    """Complete analysis response model"""

    market_data: StockData
    technical_analysis: TechnicalIndicators
    trading_signals: TradingSignals
    summary: str


# Add new structured output models for agents
class MarketDataResponse(BaseModel):
    """Structured output model for MarketDataAgent"""

    thoughts: str = Field(..., description="Analysis thoughts and reasoning")
    data: StockData = Field(..., description="Stock market data")
    status: Literal["success", "error"] = Field(
        ..., description="Status of the operation"
    )
    error_message: str | None = Field(
        None, description="Error message if status is error"
    )


class TechnicalAnalysisResponse(BaseModel):
    """Structured output model for TechnicalAnalysisAgent"""

    thoughts: str = Field(..., description="Analysis thoughts and reasoning")
    indicators: TechnicalIndicators = Field(
        ..., description="Technical analysis indicators"
    )
    signals: TradingSignals = Field(
        ..., description="Trading signals and recommendations"
    )
    summary: str = Field(..., description="Summary of the analysis")
    status: Literal["success", "error"] = Field(
        ..., description="Status of the operation"
    )
    error_message: str | None = Field(
        None, description="Error message if status is error"
    )


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


class BaseTools:
    """
    Base class for all financial analysis tools.
    Provides common functionality and logging.
    """

    def __init__(self):
        self.logger = Logger(self.__class__.__name__)

    def validate_ticker(self, ticker: str) -> None:
        """Validate ticker symbol."""
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Invalid ticker symbol provided")

    def validate_period(self, period: str) -> None:
        """Validate time period."""
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


class StockDataTools(BaseTools):
    """
    Tools for fetching and processing stock market data.
    """

    def get_data_from_yfinance(self, ticker: str, period: str) -> pd.DataFrame:
        """
        Fetch raw stock data from Yahoo Finance.

        Args:
            ticker (str): The stock ticker symbol
            period (str): The time period for data

        Returns:
            pd.DataFrame: Raw stock data from Yahoo Finance

        Raises:
            ValueError: If there's an error fetching data from Yahoo Finance
        """
        self.logger.info(
            f"External API request: service=YahooFinance, action=fetch_data, ticker={ticker}, period={period}"
        )

        try:
            # Initialize Ticker object
            stock = yf.Ticker(ticker)

            # Verify ticker exists and has data
            try:
                info = stock.info
                if not info:
                    self.logger.error(
                        f"Invalid ticker symbol: {ticker} - No data available"
                    )
                    raise ValueError(
                        f"Invalid ticker symbol: {ticker} - No data available"
                    )
            except Exception as e:
                self.logger.error(f"Error validating ticker {ticker}: {str(e)}")
                raise ValueError(f"Error validating ticker {ticker}: {str(e)}")

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

            # Verify required columns exist
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]
            if missing_columns:
                self.logger.error(
                    f"Missing required columns for {ticker}: {', '.join(missing_columns)}"
                )
                raise ValueError(
                    f"Missing required columns: {', '.join(missing_columns)}"
                )

            self.logger.info(
                f"External API response: service=YahooFinance, action=fetch_data, ticker={ticker}, status=success, rows={len(data)}"
            )
            return data

        except ValueError:
            # Re-raise ValueError as is since we've already logged it
            raise
        except Exception as e:
            error_msg = f"Unexpected error fetching data for {ticker}: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def fetch_stock_data(self, ticker: str, period: str) -> StockData:
        """
        Fetch stock data from Yahoo Finance.

        Args:
            ticker (str): The stock ticker symbol
            period (str): The time period for data

        Returns:
            StockData: Pydantic model containing stock data and metadata

        Raises:
            ValueError: If ticker is invalid or no data is found
        """
        self.logger.info(f"Fetching stock data [ticker={ticker}] [period={period}]")

        # Validate inputs using base class methods
        self.validate_ticker(ticker)
        self.validate_period(period)

        try:
            data = self.get_data_from_yfinance(ticker, period)

            # Create metadata object
            metadata = StockMetadata(
                price_change=float(data["close"].iloc[-1] - data["close"].iloc[-2]),
                volume=int(data["volume"].iloc[-1]),
                data_points=len(data),
            )

            # Create StockData object
            result = StockData(
                ticker=ticker,
                current_price=float(data["close"].iloc[-1]),
                period=period,
                data={
                    "dates": data.index.strftime("%Y-%m-%d").tolist(),
                    "open": data["open"].tolist(),
                    "high": data["high"].tolist(),
                    "low": data["low"].tolist(),
                    "close": data["close"].tolist(),
                    "volume": data["volume"].tolist(),
                },
                metadata=metadata,
            )

            self.logger.info(
                f"Data processing complete: ticker={ticker}, period={period}, data_points={len(data)}"
            )
            return result

        except ValueError:
            # Re-raise ValueError as is since we've already logged it
            raise
        except Exception as e:
            error_msg = f"Error processing stock data for {ticker}: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)


class TechnicalAnalysisTools(BaseTools):
    """
    Tools for calculating technical indicators and generating trading signals.
    """

    def calculate_technical_indicators(self, data: StockData) -> TechnicalIndicators:
        """
        Calculate technical indicators from stock data.

        Args:
            data (StockData): Pydantic model containing stock data

        Returns:
            TechnicalIndicators: Pydantic model containing technical indicators

        Raises:
            ValueError: If data is invalid or calculations fail
        """
        self.logger.info(f"Calculating technical indicators for {data.ticker}")

        def _validate_input(data):
            self.logger.info("Validating input data for required fields and length")
            required_keys = ["dates", "open", "high", "low", "close", "volume"]
            if not all(key in data.data for key in required_keys):
                self.logger.error("Missing required data fields")
                raise KeyError("Missing required data fields")
            if len(data.data["close"]) < 50:
                self.logger.error("Insufficient data points for technical analysis")
                raise ValueError("Insufficient data points for technical analysis")

        def _create_dataframe(data):
            self.logger.info("Creating DataFrame from input data")
            return pd.DataFrame(
                {
                    "close": data.data["close"],
                    "open": data.data["open"],
                    "high": data.data["high"],
                    "low": data.data["low"],
                    "volume": data.data["volume"],
                },
                index=pd.to_datetime(data.data["dates"]),
            )

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
            df["signal_line"] = df["macd"].ewm(span=9, adjust=False).mean()

        def _extract_latest_indicators(df, ticker, current_price):
            self.logger.info("Extracting latest indicator values")
            latest = df.iloc[-1]
            return TechnicalIndicators(
                ticker=ticker,
                current_price=current_price,
                sma_20=float(latest["sma_20"]),
                sma_50=float(latest["sma_50"]),
                rsi=float(latest["rsi"]),
                macd=float(latest["macd"]),
                signal_line=float(latest["signal_line"]),
            )

        try:
            _validate_input(data)
            df = _create_dataframe(data)
            _calculate_sma(df)
            _calculate_rsi(df)
            _calculate_macd(df)
            result = _extract_latest_indicators(df, data.ticker, data.current_price)
            self.logger.info(f"Successfully calculated indicators for {data.ticker}")
            return result

        except KeyError as e:
            self.logger.error(f"Missing data field: {str(e)}")
            raise ValueError(f"Missing required data field: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            raise ValueError(f"Error calculating technical indicators: {str(e)}")

    def generate_trading_signals(
        self, indicators: TechnicalIndicators
    ) -> TradingSignals:
        """
        Generate trading signals based on technical indicators.

        Args:
            indicators (TechnicalIndicators): Pydantic model containing technical indicators

        Returns:
            TradingSignals: Pydantic model containing trading signals and recommendations

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

        def analyze_macd(macd, signal_line):
            self.logger.info(f"Analyzing MACD: {macd} vs Signal Line: {signal_line}")
            if macd > signal_line:
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
            signals = []

            rsi_signal = analyze_rsi(indicators.rsi)
            if rsi_signal:
                signals.append(rsi_signal)

            macd_signal = analyze_macd(indicators.macd, indicators.signal_line)
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

            return TradingSignals(
                ticker=indicators.ticker,
                current_price=indicators.current_price,
                signals=signals,
                recommendation=recommendation,
                confidence=confidence,
                risk_level=risk_level,
            )

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


async def analyze_stock(
    ticker: str, model_client: OpenAIChatCompletionClient, tools: Tools
) -> Dict:
    logger.info(f"Starting analysis for ticker: {ticker}")

    try:
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Invalid ticker symbol provided")

        # Create agents with the provided model_client and tools
        market_data_agent = AssistantAgent(
            name="MarketDataAgent",
            description="Expert in fetching and processing financial market data",
            tools=[tools.fetch_stock_data],
            model_client=model_client,
            system_message="""You are a market data specialist. Your role is to:
            1. Fetch stock data using the fetch_stock_data tool
            2. Process and validate the data
            3. Return the data in a structured format
            
            When fetching data:
            - Always specify the ticker symbol
            - Use appropriate time periods (1d, 1wk, 1mo, 1y)
            - Verify the data is not empty
            - Check for missing values
            
            Your response must follow the MarketDataResponse format with:
            - thoughts: Your reasoning about the data
            - data: The StockData object from fetch_stock_data
            - status: "success" or "error"
            - error_message: Any error details if status is "error" """,
            output_content_type=MarketDataResponse,
            reflect_on_tool_use=True,
        )

        technical_analysis_agent = AssistantAgent(
            name="TechnicalAnalysisAgent",
            description="Expert in technical analysis and market indicators",
            tools=[
                tools.calculate_technical_indicators,
                tools.generate_trading_signals,
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
            - Provide clear, actionable recommendations
            
            Your response must follow the TechnicalAnalysisResponse format with:
            - thoughts: Your analysis reasoning
            - indicators: The TechnicalIndicators object
            - signals: The TradingSignals object
            - summary: A concise summary of findings
            - status: "success" or "error"
            - error_message: Any error details if status is "error" """,
            output_content_type=TechnicalAnalysisResponse,
            reflect_on_tool_use=True,
        )

        groupchat = SelectorGroupChat(
            participants=[market_data_agent, technical_analysis_agent],
            model_client=model_client,
            max_turns=10,
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

        # Process the final message which should be a TechnicalAnalysisResponse
        if result.messages and isinstance(
            result.messages[-1].content, TechnicalAnalysisResponse
        ):
            analysis_response = result.messages[-1].content
            return {
                "ticker": ticker,
                "analysis": {
                    "thoughts": analysis_response.thoughts,
                    "indicators": analysis_response.indicators.dict(),
                    "signals": analysis_response.signals.dict(),
                    "summary": analysis_response.summary,
                    "status": analysis_response.status,
                    "error": analysis_response.error_message,
                },
                "timestamp": datetime.now().isoformat(),
                "status": "success"
                if analysis_response.status == "success"
                else "error",
                "error": analysis_response.error_message,
            }
        else:
            error_msg = "No valid analysis response received"
            logger.error(error_msg)
            return {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": error_msg,
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

    # Initialize configuration and model client
    config = Config()
    model_client = OpenAIChatCompletionClient(
        model=config.MODEL_NAME, api_key=config.get_api_key()
    )
    logger.info("Configuration and model client initialized")

    # Initialize tools
    tools = Tools()
    logger.info("Tools initialized")

    stocks_to_analyze = ["AAPL"]

    try:
        for ticker in stocks_to_analyze:
            logger.info(f"Analyzing stock: {ticker}")
            result = await analyze_stock(ticker, model_client, tools)

            if result["status"] == "success":
                logger.info(f"Analysis completed for {ticker}")
                logger.info(f"Analysis result: {result['analysis']}")
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
