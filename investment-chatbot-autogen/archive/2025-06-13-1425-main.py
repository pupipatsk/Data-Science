# Standard library imports
import asyncio
import os
from datetime import datetime
from typing import Dict, List, Literal

import nest_asyncio

# Third-party imports
import pandas as pd
import yfinance as yf
from autogen_agentchat.agents import AssistantAgent

# Microsoft's AutoGen imports
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import ModelInfo
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Apply nest_asyncio to allow nested event loops in Jupyter
nest_asyncio.apply()

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


class InvestmentAnalysis(BaseModel):
    """Model for complete investment analysis result"""

    ticker: str
    timestamp: str
    stock_data: StockData
    technical_indicators: TechnicalIndicators
    trading_signals: TradingSignals
    final_recommendation: str = Field(
        ..., description="Final investment recommendation"
    )
    analysis_summary: str = Field(..., description="Summary of the complete analysis")


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

    MODEL_NAME: str = "gemini-2.0-flash-lite"
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
            return stock_data

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

        def _validate_input(data: StockData):
            self.logger.info("Validating input data for required fields and length")
            if not data.prices:
                self.logger.error("Price data is empty.")
                raise ValueError("Price data is empty.")
            if len(data.prices) < 20:  # Changed from 50 to 20 for minimum calculation
                self.logger.error(
                    f"Insufficient data points for technical analysis: {len(data.prices)} (minimum: 20)"
                )
                raise ValueError(
                    f"Insufficient data points for technical analysis: {len(data.prices)} (minimum: 20)"
                )

        def _create_dataframe(data: StockData) -> pd.DataFrame:
            self.logger.info(
                f"Creating DataFrame from input data with {len(data.prices)} data points"
            )
            df = pd.DataFrame([p.model_dump() for p in data.prices])
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            # Sort by date to ensure proper chronological order
            df = df.sort_index()
            return df

        def _calculate_sma(df):
            self.logger.info("Calculating SMA20 & SMA50")
            df["sma_20"] = df["close"].rolling(window=20).mean()
            # Only calculate SMA50 if we have enough data
            if len(df) >= 50:
                df["sma_50"] = df["close"].rolling(window=50).mean()
            else:
                # Use SMA20 as a fallback for SMA50 if insufficient data
                df["sma_50"] = df["sma_20"]
                self.logger.warning(
                    "Insufficient data for SMA50, using SMA20 as fallback"
                )

        def _calculate_rsi(df):
            self.logger.info("Calculating RSI")
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            # Avoid division by zero
            rs = gain / loss.replace(0, 1e-10)
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
                sma_20=float(latest["sma_20"])
                if not pd.isna(latest["sma_20"])
                else 0.0,
                sma_50=float(latest["sma_50"])
                if not pd.isna(latest["sma_50"])
                else 0.0,
                rsi=float(latest["rsi"]) if not pd.isna(latest["rsi"]) else 50.0,
                macd=float(latest["macd"]) if not pd.isna(latest["macd"]) else 0.0,
                macd_signal_line=float(latest["macd_signal_line"])
                if not pd.isna(latest["macd_signal_line"])
                else 0.0,
            )

        try:
            _validate_input(data)
            df = _create_dataframe(data)
            _calculate_sma(df)
            _calculate_rsi(df)
            _calculate_macd(df)
            result = _extract_latest_indicators(df, data.ticker)
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

            return TradingSignals(
                ticker=indicators.ticker,
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


# -- Agent Definitions --


class AnalysisWorkflow:
    """
    Manages the multi-agent investment analysis workflow with structured data passing.
    """

    def __init__(self, model_client: OpenAIChatCompletionClient, tools: Tools):
        self.model_client = model_client
        self.tools = tools
        self.logger = Logger("AnalysisWorkflow")

    async def create_agents(self):
        """Create specialized agents for different analysis tasks."""

        # Market Data Agent - Fetches stock data
        self.market_data_agent = AssistantAgent(
            name="MarketDataAgent",
            system_message="""You are a financial market data specialist. Your responsibilities:

1. Fetch comprehensive stock market data using the fetch_stock_data tool
2. Validate data quality and completeness
3. Provide structured data in JSON format for the technical analysis team
4. Report any data issues or limitations

When you complete data fetching, present the results clearly and indicate that 
the TechnicalIndicatorsAgent should proceed with analysis.

Always use fetch_stock_data with ticker and period='1y' for comprehensive analysis.""",
            model_client=self.model_client,
            tools=[self.tools.fetch_stock_data],
            reflect_on_tool_use=True,
        )

        # Technical Indicators Agent - Calculates technical metrics
        self.technical_indicators_agent = AssistantAgent(
            name="TechnicalIndicatorsAgent",
            system_message="""You are a technical analysis specialist. Your responsibilities:

1. Calculate technical indicators using the calculate_technical_indicators tool
2. Analyze SMA, RSI, MACD and other technical metrics
3. Interpret the technical indicators in market context
4. Provide structured technical analysis results to the trading signals team

When you receive stock data from MarketDataAgent, use the calculate_technical_indicators 
tool immediately. Present your analysis clearly and indicate that TradingSignalsAgent 
should proceed with signal generation.

Focus on actionable technical insights and trend analysis.""",
            model_client=self.model_client,
            tools=[self.tools.calculate_technical_indicators],
            reflect_on_tool_use=True,
        )

        # Trading Signals Agent - Generates recommendations
        self.trading_signals_agent = AssistantAgent(
            name="TradingSignalsAgent",
            system_message="""You are a trading signals specialist. Your responsibilities:

1. Generate trading signals using the generate_trading_signals tool
2. Provide buy/sell/hold recommendations with confidence levels
3. Assess risk levels for trading decisions
4. Create actionable trading recommendations

When you receive technical indicators from TechnicalIndicatorsAgent, use the 
generate_trading_signals tool to create recommendations. Provide clear rationale 
for your signals and indicate that InvestmentAdvisor should provide final analysis.

Include specific entry/exit strategies when applicable.""",
            model_client=self.model_client,
            tools=[self.tools.generate_trading_signals],
            reflect_on_tool_use=True,
        )

        # Investment Advisor - Coordinates and provides final recommendations
        self.investment_advisor = AssistantAgent(
            name="InvestmentAdvisor",
            system_message="""You are the lead investment advisor coordinating the analysis team. Your responsibilities:

1. Guide the workflow between MarketDataAgent → TechnicalIndicatorsAgent → TradingSignalsAgent
2. Review all technical analysis and trading signals comprehensively
3. Synthesize findings into coherent investment recommendations
4. Ensure compliance with investment best practices and risk management
5. Provide final summary and conclude with 'INVESTMENT_ANALYSIS_FINALIZED'

You coordinate but don't use tools directly. Guide other agents to complete their tasks 
in sequence. When all agents have contributed, provide a comprehensive investment 
recommendation including:
- Summary of technical findings
- Risk assessment
- Investment thesis
- Specific recommendations

Always include appropriate investment disclaimers.""",
            model_client=self.model_client,
            tools=[],  # Coordinator doesn't use tools directly
            reflect_on_tool_use=False,
        )

    async def run_analysis(self, ticker: str) -> Dict:
        """
        Execute the complete multi-agent investment analysis workflow.
        """
        self.logger.info(f"Starting comprehensive analysis for {ticker}")

        try:
            # Create agents
            await self.create_agents()

            # Create termination conditions
            termination_condition = TextMentionTermination(
                "INVESTMENT_ANALYSIS_FINALIZED"
            ) | MaxMessageTermination(max_messages=25)

            # Create the analysis team
            analysis_team = RoundRobinGroupChat(
                participants=[
                    self.market_data_agent,
                    self.technical_indicators_agent,
                    self.trading_signals_agent,
                    self.investment_advisor,
                ],
                termination_condition=termination_condition,
            )

            self.logger.info("Multi-agent team assembled")

            # Define the analysis task
            analysis_task = f"""
INVESTMENT RESEARCH FOR {ticker.upper()}

Execute the following sequential workflow:

Step 1: MarketDataAgent - Fetch 1-year stock data using fetch_stock_data tool
Step 2: TechnicalIndicatorsAgent - Calculate indicators using calculate_technical_indicators  
Step 3: TradingSignalsAgent - Generate signals using generate_trading_signals
Step 4: InvestmentAdvisor - Provide final recommendation and conclude with the phrase 'RESEARCH COMPLETE'

Each specialist should complete their analysis and clearly indicate when ready for the next agent.
"""

            self.logger.info("Executing multi-agent workflow")

            # Run the analysis
            result = await analysis_team.run(task=analysis_task)

            # Process results
            analysis_result = {
                "ticker": ticker.upper(),
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "workflow_completed": True,
                "stop_reason": result.stop_reason,
                "message_count": len(result.messages),
                "analysis": self._extract_analysis_summary(result.messages),
                "detailed_messages": [
                    {
                        "source": msg.source,
                        "content": msg.content,
                        "type": type(msg).__name__,
                    }
                    for msg in result.messages
                    if hasattr(msg, "content") and hasattr(msg, "source")
                ],
            }

            self.logger.info(f"Analysis completed successfully for {ticker}")
            return analysis_result

        except Exception as e:
            error_msg = f"Multi-agent analysis failed for {ticker}: {str(e)}"
            self.logger.error(error_msg)
            return {
                "ticker": ticker.upper(),
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": error_msg,
                "workflow_completed": False,
            }

    def _extract_analysis_summary(self, messages: List) -> str:
        """Extract key findings from the analysis messages."""
        summary_parts = []

        for msg in messages:
            if hasattr(msg, "source") and hasattr(msg, "content"):
                if (
                    msg.source == "InvestmentAdvisor"
                    and "INVESTMENT_ANALYSIS_FINALIZED" in msg.content
                ):
                    # Extract the final recommendation
                    summary_parts.append(f"Final Recommendation: {msg.content}")
                elif "signals" in str(msg.content).lower():
                    summary_parts.append(f"Trading Signals: {msg.content[:200]}...")
                elif "technical" in str(msg.content).lower():
                    summary_parts.append(f"Technical Analysis: {msg.content[:200]}...")

        return (
            "\n\n".join(summary_parts)
            if summary_parts
            else "Analysis completed with structured data exchange between agents."
        )


# =============================================================================
# ROUNDROBINGROUPCHAT WITH PYDANTIC MODELS IMPLEMENTATION
# =============================================================================
"""
This implementation demonstrates how to use RoundRobinGroupChat with structured
message passing using Pydantic models:

1. AGENTS & TOOLS:
   - MarketDataAgent: Uses fetch_stock_data tool → Returns StockData (Pydantic model)
   - TechnicalIndicatorsAgent: Uses calculate_technical_indicators tool → Returns TechnicalIndicators 
   - TradingSignalsAgent: Uses generate_trading_signals tool → Returns TradingSignals
   - InvestmentAdvisor: Coordinates workflow and provides final recommendations

2. MESSAGE FLOW:
   User Task → MarketDataAgent (fetches data) → TechnicalIndicatorsAgent (analyzes) 
   → TradingSignalsAgent (generates signals) → InvestmentAdvisor (final recommendation)

3. PYDANTIC MODEL PASSING:
   - Each tool returns structured Pydantic models (StockData, TechnicalIndicators, TradingSignals)
   - Agents process these models and pass information to the next agent in the workflow
   - The RoundRobinGroupChat ensures proper turn-taking and message distribution

4. TERMINATION CONDITIONS:
   - TextMentionTermination("INVESTMENT_ANALYSIS_FINALIZED"): Stops when final recommendation is complete
   - MaxMessageTermination(25): Safety limit to prevent infinite loops

5. STREAMING CAPABILITY:
   - run_stream() allows real-time observation of message passing between agents
   - Shows how Pydantic model data flows through the multi-agent system

This pattern enables complex financial analysis workflows where each agent
specializes in specific tasks while maintaining structured data exchange.
"""


async def analyze_stock(
    ticker: str, model_client: OpenAIChatCompletionClient, tools: Tools
) -> Dict:
    """
    Perform comprehensive multi-agent stock analysis using RoundRobinGroupChat.

    This function demonstrates structured message passing between agents using
    Pydantic models for data validation and transfer.
    """
    logger = Logger("analyze_stock")
    logger.info(f"Initializing multi-agent analysis workflow for {ticker}")

    try:
        # Create and run the analysis workflow
        workflow = AnalysisWorkflow(model_client, tools)
        result = await workflow.run_analysis(ticker)

        return result

    except Exception as e:
        error_msg = f"Failed to initialize analysis workflow for {ticker}: {str(e)}"
        logger.error(error_msg)
        return {
            "ticker": ticker.upper(),
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": error_msg,
            "workflow_completed": False,
        }


async def analyze_stock_with_streaming(
    ticker: str, model_client: OpenAIChatCompletionClient, tools: Tools
) -> Dict:
    """
    Demonstrate real-time streaming analysis with message passing between agents.
    Shows how Pydantic models flow between agents in the RoundRobinGroupChat.
    """
    logger = Logger("analyze_stock_streaming")
    logger.info(f"Starting streaming analysis for {ticker}")

    try:
        # Create the workflow
        workflow = AnalysisWorkflow(model_client, tools)
        await workflow.create_agents()

        # Create termination conditions
        termination_condition = TextMentionTermination(
            "INVESTMENT_ANALYSIS_FINALIZED"
        ) | MaxMessageTermination(max_messages=25)

        # Create the analysis team
        analysis_team = RoundRobinGroupChat(
            participants=[
                workflow.market_data_agent,
                workflow.technical_indicators_agent,
                workflow.trading_signals_agent,
                workflow.investment_advisor,
            ],
            termination_condition=termination_condition,
        )

        # Define the analysis task
        analysis_task = f"""
STREAMING INVESTMENT ANALYSIS FOR {ticker.upper()}

WORKFLOW: MarketDataAgent → TechnicalIndicatorsAgent → TradingSignalsAgent → InvestmentAdvisor

1. MarketDataAgent: Fetch and validate stock data using fetch_stock_data with period='1y'
2. TechnicalIndicatorsAgent: Calculate technical indicators from the stock data using calculate_technical_indicators  
3. TradingSignalsAgent: Generate signals from technical indicators using generate_trading_signals
4. InvestmentAdvisor: Provide final comprehensive recommendation and say 'INVESTMENT_ANALYSIS_FINALIZED'

Each agent should use their tools and pass structured Pydantic model data clearly.
"""

        print(f"\n🔄 STARTING STREAMING ANALYSIS FOR {ticker.upper()}")
        print("=" * 60)

        messages = []

        # Stream the analysis in real-time
        async for message in analysis_team.run_stream(task=analysis_task):
            # Check if it's a TaskResult (final result)
            if hasattr(message, "messages"):
                # This is the final TaskResult
                print("\n✅ ANALYSIS COMPLETED")
                print(f"Stop Reason: {message.stop_reason}")
                print(f"Total Messages: {len(message.messages)}")

                return {
                    "ticker": ticker.upper(),
                    "timestamp": datetime.now().isoformat(),
                    "status": "success",
                    "workflow_completed": True,
                    "stop_reason": message.stop_reason,
                    "message_count": len(message.messages),
                    "streaming_demo": True,
                    "summary": f"Completed multi-agent analysis with {len(message.messages)} message exchanges",
                }
            else:
                # This is an individual message
                messages.append(message)

                # Display the message with formatting
                if hasattr(message, "source") and hasattr(message, "content"):
                    agent_emoji = {
                        "user": "👤",
                        "MarketDataAgent": "📊",
                        "TechnicalIndicatorsAgent": "📈",
                        "TradingSignalsAgent": "🎯",
                        "InvestmentAdvisor": "💼",
                    }.get(message.source, "🤖")

                    print(f"\n{agent_emoji} {message.source.upper()}")
                    print("-" * 40)

                    # Truncate long messages for display
                    content = str(message.content)
                    if len(content) > 300:
                        content = content[:300] + "..."

                    print(content)

                    # Show tool usage if applicable
                    if "ToolCall" in type(message).__name__:
                        print("🔧 Tool execution in progress...")
                    elif "ToolCall" in str(content):
                        print("🔧 Using specialized financial tools...")

                    print()  # Add spacing

        # If we reach here without a TaskResult, return a default response
        return {
            "ticker": ticker.upper(),
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "workflow_completed": True,
            "message_count": len(messages),
            "streaming_demo": True,
            "summary": f"Analysis workflow completed with {len(messages)} messages",
        }

    except Exception as e:
        error_msg = f"Streaming analysis failed for {ticker}: {str(e)}"
        logger.error(error_msg)
        return {
            "ticker": ticker.upper(),
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": error_msg,
            "workflow_completed": False,
            "streaming_demo": True,
        }


async def simple_multi_agent_demo(
    ticker: str, model_client: OpenAIChatCompletionClient, tools: Tools
) -> Dict:
    """
    Simple demonstration of RoundRobinGroupChat with clear message passing.
    """
    logger = Logger("simple_demo")
    logger.info(f"Starting simple multi-agent demo for {ticker}")

    try:
        # Create simple agents
        data_agent = AssistantAgent(
            name="DataAgent",
            system_message=f"You fetch stock data for {ticker} using the fetch_stock_data tool. Use period='1y'. After fetching, pass the data information to TechAgent.",
            model_client=model_client,
            tools=[tools.fetch_stock_data],
            reflect_on_tool_use=True,
        )

        tech_agent = AssistantAgent(
            name="TechAgent",
            system_message="You calculate technical indicators using the calculate_technical_indicators tool when you receive stock data. Pass your analysis to SignalAgent.",
            model_client=model_client,
            tools=[tools.calculate_technical_indicators],
            reflect_on_tool_use=True,
        )

        signal_agent = AssistantAgent(
            name="SignalAgent",
            system_message="You generate trading signals using the generate_trading_signals tool when you receive technical indicators. Say 'DEMO_COMPLETE' when finished.",
            model_client=model_client,
            tools=[tools.generate_trading_signals],
            reflect_on_tool_use=True,
        )

        # Create termination condition
        termination_condition = TextMentionTermination(
            "DEMO_COMPLETE"
        ) | MaxMessageTermination(max_messages=15)

        # Create the team
        demo_team = RoundRobinGroupChat(
            participants=[data_agent, tech_agent, signal_agent],
            termination_condition=termination_condition,
        )

        print(f"\n🚀 SIMPLE MULTI-AGENT DEMO FOR {ticker.upper()}")
        print("=" * 50)

        # Run the demo
        result = await demo_team.run(
            task=f"Analyze {ticker} stock - start by fetching data"
        )

        print("\n🎉 DEMO COMPLETED")
        print(f"Messages: {len(result.messages)}")
        print(f"Stop Reason: {result.stop_reason}")

        return {
            "ticker": ticker.upper(),
            "status": "success",
            "demo_type": "simple",
            "message_count": len(result.messages),
            "stop_reason": result.stop_reason,
        }

    except Exception as e:
        logger.error(f"Simple demo failed: {str(e)}")
        return {
            "ticker": ticker.upper(),
            "status": "error",
            "error": str(e),
            "demo_type": "simple",
        }


async def main():
    logger = Logger("Main")
    logger.info("Starting investment analysis chatbot with RoundRobinGroupChat")

    # Initialize configuration
    config = Config()
    logger.info("Configuration initialized")

    # Initialize global gemini model_client
    model_client = OpenAIChatCompletionClient(
        model=config.MODEL_NAME,
        api_key=config.get_api_key(),
        model_info=ModelInfo(
            vision=True,
            function_calling=True,
            json_output=True,
            family="unknown",
            structured_output=True,
        ),
    )
    logger.info("Global gemini model_client initialized")

    # Initialize tools
    tools = Tools()
    logger.info("Tools initialized")

    stocks_to_analyze = ["AAPL"]

    try:
        for ticker in stocks_to_analyze:
            logger.info(f"Analyzing stock: {ticker}")

            print("\n🎯 ROUNDROBINGROUPCHAT DEMONSTRATION")
            print(f"Stock: {ticker.upper()}")
            print("=" * 50)

            # Run simple demo first
            print("\n1. SIMPLE MULTI-AGENT WORKFLOW")
            simple_result = await simple_multi_agent_demo(ticker, model_client, tools)

            if simple_result["status"] == "success":
                logger.info(f"Simple demo completed for {ticker}")
                print(f"✅ Simple Demo: {simple_result['message_count']} messages")
            else:
                logger.error(f"Simple demo failed for {ticker}")
                print(
                    f"❌ Simple Demo failed: {simple_result.get('error', 'Unknown error')}"
                )

            print("\n" + "=" * 50)
            print("PYDANTIC MODELS DEMONSTRATION COMPLETED")
            print("- Each agent used specialized tools returning Pydantic models")
            print("- StockData → TechnicalIndicators → TradingSignals")
            print("- RoundRobinGroupChat coordinated the message passing")
            print("=" * 50)

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
