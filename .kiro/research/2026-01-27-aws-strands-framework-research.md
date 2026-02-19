# Research Report: AWS Strands Framework

This research provides comprehensive answers to user questions about AWS Strands installation, API key configuration, MCP integration, and basic usage patterns for connecting to MCP servers.

## Research Scope
- Research Type: Architecture/Best Practices
- Queries Used: 15/20
- Coverage: Comprehensive
- Questions Investigated: Installation methods, API configuration, MCP integration, basic usage patterns

## Question 1: Installation - How do I install AWS Strands?

**Answer:** Install using `pip install strands-agents` (not `aws-strands`)

AWS Strands Agents is installed via the Python Package Index using the package name `strands-agents` [1]. The installation process follows standard Python package management practices, requiring Python 3.10+ [1]. The basic installation command is:

```bash
pip install strands-agents
```

For development purposes, additional packages are available that extend functionality. The `strands-agents-tools` package provides pre-built tools for common tasks, while `strands-agents-builder` offers an agent that helps build custom Strands agents and tools [1]. These can be installed together:

```bash
pip install strands-agents strands-agents-tools strands-agents-builder
```

The framework supports optional dependencies for different model providers. For example, to use OpenAI models, install with: `pip install 'strands-agents[openai]'` [7]. For comprehensive provider support, use: `pip install 'strands-agents[all]'` [7].

**Virtual Environment Setup:** The official documentation recommends creating a virtual environment before installation [1]:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install strands-agents
```

## Question 2: API Key Configuration - Does Strands use OpenAI API directly or AWS Bedrock?

**Answer:** Strands defaults to AWS Bedrock but supports direct OpenAI API usage and multiple other providers with flexible configuration.

AWS Strands Agents employs a model-agnostic architecture that supports multiple LLM providers [2]. By default, agents use Amazon Bedrock with the Claude 4 Sonnet model in the US Oregon (us-west-2) region [1]. However, the framework provides flexible model provider configuration allowing direct integration with OpenAI and other services.

**Default Bedrock Configuration:** When using the default setup, Strands requires AWS credentials configured through standard methods [1]:
- Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- AWS credentials file via `aws configure`
- IAM roles for AWS services (EC2, ECS, Lambda)

Model access must be enabled in the Amazon Bedrock console following AWS documentation [1].

**OpenAI Direct Integration:** For direct OpenAI API usage, Strands provides dedicated model providers [7]:

```python
from strands import Agent
from strands.models.openai import OpenAIModel

model = OpenAIModel(
    client_args={"api_key": "<KEY>"},
    model_id="gpt-4o",
    params={"max_tokens": 1000, "temperature": 0.7}
)

agent = Agent(model=model)
```

**Model Provider Flexibility:** The framework supports over 10 different model providers including Anthropic, Gemini, LiteLLM, Ollama, MistralAI, and others [7]. This allows developers to switch between providers without changing core agent code, providing significant deployment flexibility.

**API Key Management:** API keys are configured at the model provider level rather than globally. For OpenAI, keys can be set via environment variables (`OPENAI_API_KEY`) or passed directly in the model configuration [7]. This approach enables multi-provider setups where different agents use different LLM services within the same application.

## Question 3: MCP Integration - Does Strands have native MCP support?

**Answer:** Yes, Strands has comprehensive native MCP (Model Context Protocol) support with built-in client capabilities and multiple transport options.

AWS Strands Agents includes first-class support for the Model Context Protocol (MCP), an open protocol that standardizes how applications provide context to Large Language Models [8]. This integration enables agents to connect to external MCP servers and use their tools seamlessly.

**Native MCP Client:** Strands provides a built-in `MCPClient` class that implements the `ToolProvider` interface, allowing direct usage in agent constructors with automatic lifecycle management [8]:

```python
from mcp import stdio_client, StdioServerParameters
from strands import Agent
from strands.tools.mcp import MCPClient

# Create MCP client with stdio transport
mcp_client = MCPClient(lambda: stdio_client(
    StdioServerParameters(
        command="uvx",
        args=["awslabs.aws-documentation-mcp-server@latest"]
    )
))

# Pass MCP client directly to agent
agent = Agent(tools=[mcp_client])
```

**Multiple Transport Support:** The framework supports three primary MCP transport mechanisms [8]:

1. **Standard I/O (stdio):** For command-line tools and local processes
2. **Streamable HTTP:** For HTTP-based MCP servers with authentication support
3. **Server-Sent Events (SSE):** For real-time communication with MCP servers

**Multi-Server Integration:** Strands enables connecting to multiple MCP servers simultaneously, combining tools from different sources [8]:

```python
# Multiple MCP clients in single agent
agent = Agent(tools=[mcp_client_1, mcp_client_2])
```

**Advanced Features:** The MCP integration includes tool filtering, name prefixing to prevent conflicts, and support for AWS IAM authentication through the `mcp-proxy-for-aws` package [8]. The framework also supports MCP elicitation for interactive tool execution requiring user confirmation.

**Integration Approaches:** Strands offers both managed integration (recommended) where the framework handles connection lifecycle automatically, and manual context management for explicit control over MCP sessions [8].

## Question 4: Basic Usage - What's the minimal code to create a Strands agent?

**Answer:** The minimal code requires just 3 lines: import Agent, create instance, and call with a message.

AWS Strands Agents follows a "model-driven" approach that emphasizes simplicity and minimal code requirements [2]. The framework leverages the reasoning capabilities of modern LLMs to handle planning and tool usage autonomously, eliminating the need for complex workflow definitions.

**Minimal Agent Example:**

```python
from strands import Agent

# Create agent with default settings
agent = Agent()

# Ask the agent a question
agent("Tell me about agentic AI")
```

This three-line example creates a fully functional agent using default configurations (Amazon Bedrock with Claude 4 Sonnet) [13].

**Framework Philosophy:** Unlike frameworks that require developers to define complex workflows, Strands embraces the native capabilities of state-of-the-art language models to plan, reason, call tools, and reflect on their actions [2]. Developers simply define an agent's goal via a prompt and provide a list of tools, while the underlying LLM handles orchestration.

**Comparison to Other Frameworks:** Strands differs significantly from LangChain and similar frameworks in its approach [15]. While LangChain requires explicit workflow definition and chain construction, Strands relies on model reasoning for autonomous task execution. This model-driven approach reduces development complexity but requires trust in the LLM's planning capabilities.

**Tool Integration:** Adding tools follows the same minimal pattern:

```python
from strands import Agent, tool
from strands_tools import calculator, current_time

@tool
def custom_function(input: str) -> str:
    """Custom tool description"""
    return f"Processed: {input}"

agent = Agent(tools=[calculator, current_time, custom_function])
agent("What time is it and what's 2+2?")
```

**Production Considerations:** While the minimal code approach accelerates prototyping, production deployments require additional configuration for observability, error handling, and deployment infrastructure [15]. Strands provides comprehensive deployment options including AWS Lambda, EKS, EC2, and Bedrock AgentCore integration.

## Synthesis Across Questions

The research reveals AWS Strands as a deliberately simplified framework that prioritizes developer experience through minimal code requirements while maintaining enterprise-grade capabilities. The framework's architecture demonstrates several key design principles:

**Model-Driven Philosophy:** Strands distinguishes itself from workflow-based frameworks like LangChain by embracing LLM reasoning capabilities for autonomous task orchestration. This approach reduces boilerplate code but requires careful prompt engineering and model selection.

**Provider Agnosticism:** Despite AWS origins, Strands maintains true provider neutrality, supporting over 10 different LLM providers with consistent APIs. This flexibility enables hybrid deployments and vendor diversification strategies.

**Standards Adoption:** The comprehensive MCP support positions Strands as forward-looking in agent interoperability. MCP integration enables dynamic tool discovery and standardized external service connections, reducing custom integration overhead.

**Enterprise Readiness:** While maintaining simplicity, Strands provides production-grade features including observability, multi-agent patterns, and comprehensive AWS service integration. The framework bridges the gap between rapid prototyping and enterprise deployment.

## Limitations and Gaps

Several areas require consideration when evaluating Strands for production use:

**Community Maturity:** As a newer framework (~3.2k GitHub stars), Strands has a smaller community compared to established alternatives like LangChain (100k+ stars) [15]. This affects available resources, third-party integrations, and talent availability.

**Infrastructure Responsibility:** Unlike managed services, Strands requires users to handle hosting, scaling, and monitoring infrastructure [15]. Organizations must invest in operational capabilities for production deployments.

**Model Dependency:** The model-driven approach creates strong dependency on LLM reasoning quality. Poor model selection or prompt engineering can significantly impact agent reliability and predictability.

**Observability Gaps:** While Strands provides basic observability features, it lacks native integration with popular tools like LangFuse, potentially requiring custom monitoring solutions [15].

## Conclusions

AWS Strands Agents represents a compelling approach to AI agent development that prioritizes simplicity without sacrificing capability. The framework's minimal code requirements, comprehensive MCP support, and provider flexibility make it particularly suitable for:

- **Rapid Prototyping:** The three-line agent creation enables fast proof-of-concept development
- **AWS-Centric Organizations:** Native AWS integration provides seamless cloud deployment paths
- **Multi-Provider Strategies:** Model-agnostic architecture supports vendor diversification
- **Standards-Forward Development:** MCP integration future-proofs agent architectures

For organizations seeking to balance development speed with production readiness, Strands offers a pragmatic middle ground between simple prototyping frameworks and complex enterprise solutions. However, teams should carefully evaluate community support, operational requirements, and observability needs before committing to production deployments.

## Sources

### Primary Sources
[1] Strands Agents Quickstart Guide - https://strandsagents.com/1.0.x/documentation/docs/user-guide/quickstart/
    Relevance: Official installation instructions, basic setup, and configuration requirements

[2] AWS Prescriptive Guidance: Strands Agents - https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-frameworks/strands-agents.html
    Relevance: Comprehensive framework overview, key features, and architectural principles

[7] Strands Model Providers Documentation - https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/
    Relevance: Complete model provider configuration, API key management, and multi-provider setup

[8] Strands MCP Tools Documentation - https://strandsagents.com/latest/documentation/docs/user-guide/concepts/tools/mcp-tools/
    Relevance: Comprehensive MCP integration guide, transport options, and usage patterns

### Supporting Sources
[13] Strands Agents SDK Homepage - https://strandsagents.com/1.0.x/documentation/docs/
    Relevance: Basic usage examples and hello world implementation

[15] Battle of AI Agent Frameworks Comparison - https://sabootcamp.kit.com/posts/battle-of-the-ai-agent-frameworks
    Relevance: Framework comparison, strengths/weaknesses analysis, and production considerations