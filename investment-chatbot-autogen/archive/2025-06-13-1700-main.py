import asyncio
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import yfinance as yf

# Microsoft's AutoGen imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_core.models import ModelInfo
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field


# -- Settings --
class Logger:
    """Logger class for handling different types of log messages."""

    LOG_FILE: str = "log.log"

    def __init__(self, module_name: str) -> None:
        """
        Initialize logger with module name.

        Args:
            module_name: Name of the module using the logger
        """
        self.module_name: str = module_name

    def _write_log(self, level: str, message: str) -> None:
        """
        Write a log message to both console and file.

        Args:
            level: Log level (INFO, WARNING, ERROR)
            message: Log message
        """
        timestamp: str = datetime.now().isoformat()
        log_entry: str = f"[{timestamp}][{level}][{self.module_name}] {message}"
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


class ApiCredentialsManager:
    """
    Configuration class for managing API credentials.
    """

    API_KEY_NAME: str = "GEMINI_API_KEY"

    def __init__(self) -> None:
        """Initialize configuration and load environment variables."""
        self.logger: Logger = Logger("ApiCredentialsManager")
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
        api_key: Optional[str] = os.getenv(self.API_KEY_NAME)
        if not api_key:
            self.logger.error(f"{self.API_KEY_NAME} not set")
            raise ValueError(f"{self.API_KEY_NAME} environment variable not set.")
        self.logger.info(f"{self.API_KEY_NAME} access successful")
        return api_key


def get_stock_data_from_yfinance(ticker: str) -> Dict[str, Any]:
    """
    Retrieve stock data from Yahoo Finance for a specific ticker,
    including key performance indicators.

    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL' for Apple).

    Returns:
        Dictionary containing stock data and key performance indicators
    """
    logger: Logger = Logger("get_stock_data_from_yfinance")
    try:
        logger.info(f"ExternalAPI, service=yfinance, type=request, ticker={ticker}")
        try:
            stock: yf.Ticker = yf.Ticker(ticker)
        except Exception as e:
            logger.error(f"Failed to access yfinance API for {ticker}: {e}")
            return {}
        logger.info(f"ExternalAPI, service=yfinance, type=response, ticker={ticker}")

        info: Dict[str, Any] = stock.info

        return {
            "company": info.get("shortName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap_b": round(info.get("marketCap", 0) / 1_000_000_000, 2),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "forward_pe": info.get("forwardPE", "N/A"),
            "eps_ttm": info.get("trailingEps", "N/A"),
            "dividend_yield": round(info.get("dividendYield", 0) * 100, 2)
            if info.get("dividendYield")
            else "N/A",
            "profit_margin": round(info.get("profitMargins", 0) * 100, 2)
            if info.get("profitMargins")
            else "N/A",
            "roe": round(info.get("returnOnEquity", 0) * 100, 2)
            if info.get("returnOnEquity")
            else "N/A",
            "debt_to_equity": info.get("debtToEquity", "N/A"),
            "beta_5y": info.get("beta", "N/A"),
            "week_high_52": info.get("fiftyTwoWeekHigh", "N/A"),
            "week_low_52": info.get("fiftyTwoWeekLow", "N/A"),
            "analyst_rating": info.get("recommendationKey", "N/A").capitalize()
            if info.get("recommendationKey")
            else "N/A",
            "target_price": info.get("targetMeanPrice", "N/A"),
        }

    except Exception as e:
        logger.error(f"Error retrieving data for {ticker}: {e}")
        return {}


# -- Pydantic models --


class Metrics(BaseModel):
    """Model for representing a relevant metric with its rating."""

    metric: str = Field(description="Name of the metric")
    rating: int = Field(description="Rating from 0-10, where 10 is best")


class InvestmentAnalysisResponse(BaseModel):
    """Model for representing a complete investment analysis."""

    reasoning: str = Field(
        description="Detailed reasoning for the investment decision based on Warren Buffett's principles"
    )
    rating: int = Field(
        description="Overall investment rating from 0-10, where 10 is best"
    )
    relevant_metrics: List[Metrics] = Field(
        description="List of relevant metrics with individual ratings"
    )
    improvement_requirements: str = Field(
        description="Areas where the company needs to improve"
    )


# -- Workflow --
class Workflow:
    def __init__(self, ticker: str) -> None:
        """
        Initialize the Workflow for a stock analysis.

        Args:
            ticker: The stock ticker symbol (e.g., 'AAPL' for Apple)
        """
        self.logger: Logger = Logger("Workflow")
        self.ticker: str = ticker
        self.get_stock_data_tool: FunctionTool = FunctionTool(
            get_stock_data_from_yfinance,
            description="Retrieve financial data for a stock using yfinance API",
            strict=True,
        )

    async def run(self) -> Dict[str, Any]:
        """
        Run the stock analysis based on Warren Buffett's principles.

        Returns:
            Dictionary with the analysis results
        """
        self.logger.info(f"Analyzing stock: {self.ticker}")

        # Initialize model_client
        self.logger.info("Initializing model client")
        model_client: OpenAIChatCompletionClient = OpenAIChatCompletionClient(
            model="gemini-2.0-flash-lite",
            model_info=ModelInfo(
                vision=True,
                function_calling=True,
                json_output=True,
                family="unknown",
                structured_output=True,
            ),
            api_key=ApiCredentialsManager().get_api_key(),
        )

        system_message: str = f"""You are an expert investment analyst who follows Warren Buffett's 
            investment principles to evaluate stocks. You believe in value investing and 
            long-term perspectives.
            
            Here are Warren Buffett's investment principles:
            Value investing: Buffett looks for undervalued companies - those trading below their intrinsic value. He aims to buy great businesses at fair prices, rather than fair businesses at great prices.
            Long-term perspective: He invests with a long-term outlook, often holding stocks for decades. This aligns with his belief that the stock market is unpredictable in the short term but tends to rise over the long term.
            Understanding the business: Buffett only invests in companies and industries he thoroughly understands. This principle helps him better assess the company's potential and risks.
            Competitive advantage: He looks for companies with a strong "economic moat" - a sustainable competitive advantage that protects the company from competitors.
            Quality management: Buffett places high importance on honest, competent management teams that act in the best interests of shareholders.
            Financial health: He prefers companies with strong balance sheets, consistent earnings, high return on equity, and low debt.
            Predictable earnings: Buffett favors businesses with steady and predictable earnings rather than those in volatile or cyclical industries.
            Margin of safety: He always aims to buy stocks with a significant margin of safety - the difference between the intrinsic value of a stock and its market price.
            Dividend policy: While not a strict requirement, Buffett often favors companies that pay dividends and have a history of increasing them.
            Circle of competence: He sticks to investments within his "circle of competence" - areas where he has deep knowledge and expertise.
            
            Your task is to analyze the stock with ticker {self.ticker} based on Warren Buffett's investment principles.
            Use the get_stock_data tool to retrieve the financial data for this stock.
            
            After analyzing the stock, provide a structured investment analysis that includes:
            1. Detailed reasoning for your investment decision based on Warren Buffett's principles
            2. Overall investment rating from 0-10, where 10 is best
            3. List of relevant metrics with individual ratings from 0-10
            4. Areas where the company needs to improve
            
            Your response must be a valid JSON object that strictly follows the InvestmentAnalysisResponse schema below. 
            Do not include any text or explanation outside the JSON object. 
            Ensure all required fields are present and use clear, concise language.

            InvestmentAnalysisResponse schema: {InvestmentAnalysisResponse.model_json_schema()}
            """

        self.logger.info("Initializing InvestmentAnalystAgent")
        analyst: AssistantAgent = AssistantAgent(
            name="InvestmentAnalyst",
            description="An investment analyst who follows Warren Buffett's principles",
            model_client=model_client,
            system_message=system_message,
            tools=[self.get_stock_data_tool],
            reflect_on_tool_use=True,
        )

        self.logger.info("Creating request_message")
        request_message: TextMessage = TextMessage(
            content=f"Analyze {self.ticker} stock and provide a structured investment analysis based on Warren Buffett's principles.",
            source="user",
        )

        self.logger.info("Sending request_message to Agent")
        response_message: Response = await analyst.on_messages(
            [request_message],
            cancellation_token=CancellationToken(),
        )
        self.logger.info("Received response_message from Agent")

        final_content: str = response_message.chat_message.content  # ```json...```
        try:
            json_match: Optional[re.Match[str]] = re.search(
                r"```json\s*(.*?)\s*```", final_content, re.DOTALL
            )
            if json_match:
                json_str: str = json_match.group(1)
                parsed_data: Dict[str, Any] = json.loads(json_str)
                self.logger.info(
                    f"Successfully extracted and analyzed JSON for {self.ticker}"
                )
                return parsed_data
        except Exception as json_error:
            self.logger.error(f"Failed to extract JSON: {json_error}")
            return {
                "reasoning": "FAILED",
                "rating": -1,
                "relevant_metrics": [],
                "improvement_requirements": "FAILED",
            }  # Fallback

    def sync_run(self) -> Dict[str, Any]:
        """
        Synchronous wrapper for the run method.

        Returns:
            Dictionary with the analysis results
        """
        self.logger.info("sync_run() called")
        return asyncio.run(self.run())


def main(ticker: str = "AAPL") -> None:
    """
    Main function to run the stock analysis.

    Args:
        ticker: The stock ticker symbol to analyze (default: "AAPL")
    """
    logger: Logger = Logger("main")
    start_time: float = time.time()

    # Create and run the workflow
    logger.info("Initializing workflow")
    workflow: Workflow = Workflow(ticker=ticker)
    logger.info("Running workflow")
    result: Dict[str, Any] = workflow.sync_run()

    # Print JSON version
    result_json: str = json.dumps(result, indent=4)
    print(f"\nJSON Result:\n{result_json}")

    # Print human-readable version
    def print_human_readable_result(result: Dict[str, Any]) -> None:
        print("\n" + "=" * 50)
        print(f"INVESTMENT ANALYSIS FOR {ticker}")
        print("=" * 50)

        print("\nREASONING:")
        print(result["reasoning"])

        print(f"\nOVERALL RATING: {result['rating']}/10")

        print("\nRELEVANT metrics:")
        for metric in result["relevant_metrics"]:
            print(f"- {metric['metric']}: {metric['rating']}/10")

        print("\nIMPROVEMENT REQUIREMENTS:")
        print(result["improvement_requirements"])

        print("\n" + "=" * 50)

    print_human_readable_result(result)

    end_time: float = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
