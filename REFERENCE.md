# DSPy.rb - Comprehensive Reference

> Build LLM apps like you build software. Type-safe, modular, testable.

DSPy.rb brings software engineering best practices to LLM development. Instead of tweaking prompts, you define what you want with Ruby types and let DSPy handle the rest.

## Table of Contents

1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Core Concepts](#core-concepts)
4. [Signatures](#signatures)
5. [Modules](#modules)
6. [Predictors](#predictors)
7. [Rich Types](#complex-types)
8. [Multimodal Support](#multimodal-support)
9. [Agent Systems](#agent-systems)
10. [Memory Systems](#memory-systems)
11. [Toolsets](#toolsets)
12. [Optimization](#optimization)
13. [Production Features](#production-features)
14. [Testing Strategies](#testing-strategies)
15. [API Reference](#api-reference)
16. [Integration Guides](#integration-guides)
17. [Examples](#examples)

## Overview

DSPy.rb is a Ruby framework for building language model applications with programmatic prompts. It provides:

- **Type-safe signatures** - Define inputs/outputs with Sorbet types
- **Modular components** - Compose and reuse LLM logic
- **Automatic optimization** - Use data to improve prompts, not guesswork
- **Production-ready** - Built-in observability, testing, and error handling

### Key Features

- **Provider Support**: OpenAI, Anthropic, Google Gemini, Ollama (via OpenAI compatibility)
- **Type Safety**: Sorbet integration throughout
- **Automatic JSON Extraction**: Provider-optimized strategies
- **Composable Modules**: Chain, compose, and reuse
- **Multimodal Support**: Text and image inputs with vision models via raw chat (signature wiring planned)
- **Agent Systems**: ReAct (core), CodeAct (`dspy-code_act`), and custom agents
- **Memory & State**: Persistent memory for stateful applications
- **Observability**: Automatic APM integration, token tracking, performance monitoring

### Provider Compatibility

| Feature | OpenAI | Anthropic | Gemini | Ollama |
|---------|--------|-----------|--------|--------|
| Text Generation | ✅ | ✅ | ✅ | ✅ |
| Structured Output | ✅ | ✅ | ✅ | ✅ |
| Vision (Raw Chat) | ✅ | ✅ | ✅ | ❌ |
| Vision (Signatures)  | ✅ | ✅ | ✅ | ❌ |
| Image URLs | ✅ | ❌ | ❌ | ❌ |
| Image Base64 | ✅ | ✅ | ✅ | ❌ |
| Tool Calling |  ✅ | ✅ | ✅ | Varies |

### Current Limitations

- **Streaming**: Supported via block streaming on OpenAI, Anthropic, and Gemini adapters; modules return concatenated content only (no token-by-token callbacks).
- **Function/Tool Calling**: Anthropic adapter accepts `tools:`; OpenAI and Gemini adapters do not yet expose tool specs.
- **Image URLs**: Only OpenAI supports direct URL references.
- **Local Models**: Limited multimodal support through Ollama.
- **Batch Processing**: Single request processing only.

## Installation & Setup

### Requirements

- Ruby 3.3 or higher
- Bundler

### Installation

Add to your Gemfile:

```ruby
gem 'dspy'
```

Then run:

```bash
bundle install
```

### Provider Adapter Gems

Add the adapter gems that match the providers you call so DSPy only pulls the SDKs you actually use:

```ruby
# Gemfile
gem 'dspy'
gem 'dspy-openai'    # OpenAI, OpenRouter, Ollama
gem 'dspy-anthropic' # Claude
gem 'dspy-gemini'    # Gemini
```

Each adapter gem already depends on the official SDK (`openai`, `anthropic`, `gemini-ai`), so you don't need to add those manually. DSPy auto-loads adapters when the gem is present—no extra `require` needed. See the adapter READMEs for details:

- [OpenAI / OpenRouter / Ollama adapters](https://github.com/vicentereig/dspy.rb/blob/main/lib/dspy/openai/README.md)
- [Anthropic adapters](https://github.com/vicentereig/dspy.rb/blob/main/lib/dspy/anthropic/README.md)
- [Gemini adapters](https://github.com/vicentereig/dspy.rb/blob/main/lib/dspy/gemini/README.md)

### Basic Configuration

```ruby
require 'dspy'

# Configure with OpenAI
DSPy.configure do |c|
  c.lm = DSPy::LM.new('openai/gpt-4o-mini', api_key: ENV['OPENAI_API_KEY'])
end

# Or configure with Anthropic
DSPy.configure do |c|
  c.lm = DSPy::LM.new('anthropic/claude-3-sonnet', api_key: ENV['ANTHROPIC_API_KEY'])
end

# Or configure with Google Gemini
DSPy.configure do |c|
  c.lm = DSPy::LM.new('gemini/gemini-1.5-pro', api_key: ENV['GEMINI_API_KEY'])
end

# Or use Ollama for local models
DSPy.configure do |c|
  c.lm = DSPy::LM.new('ollama/llama3.2')  # No API key needed for local
end
```

### Environment Variables

```bash
# LLM API Keys
export OPENAI_API_KEY=sk-your-key-here
export ANTHROPIC_API_KEY=sk-ant-your-key-here
export GEMINI_API_KEY=your-gemini-key

# Optional: Observability
export OTEL_SERVICE_NAME=my-dspy-app
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export LANGFUSE_SECRET_KEY=sk_your_key
export LANGFUSE_PUBLIC_KEY=pk_your_key
export NEW_RELIC_LICENSE_KEY=your_license_key
```

### Advanced Configuration

```ruby
DSPy.configure do |c|
  # Language Model
  c.lm = DSPy::LM.new('openai/gpt-4o-mini',
    api_key: ENV['OPENAI_API_KEY'],
    temperature: 0.7,
    max_tokens: 2000
  )

  # Observability
  c.logger = Dry.Logger(:dspy, formatter: :json) do |logger|
    logger.add_backend(level: :info, stream: $stdout)
  end
end
```

## Core Concepts

### 1. Signatures

Signatures define the interface between your application and language models:

```ruby
class EmailClassifier < DSPy::Signature
  description "Classify customer support emails by category and priority"

  class Priority < T::Enum
    enums do
      Low = new('low')
      Medium = new('medium')
      High = new('high')
      Urgent = new('urgent')
    end
  end

  input do
    const :email_content, String
    const :sender, String
  end

  output do
    const :category, String
    const :priority, Priority
    const :confidence, Float
  end
end
```

### 2. Modules

Modules provide reusable LLM components:

- **DSPy::Module** - Base class for custom modules
- **Per-instance configuration** - Each module can have its own LM
- **Composability** - Combine modules for complex workflows

### 3. Predictors

Built-in predictors for different reasoning patterns:

- **Predict** - Basic LLM calls
- **ChainOfThought** - Step-by-step reasoning
- **ReAct** - Tool-using agents
- **CodeAct** - Dynamic code generation (install the `dspy-code_act` gem)

### 4. Optimization

Improve accuracy with data:

- **MIPROv2** - Advanced multi-prompt optimization with bootstrap sampling, instruction generation, and Bayesian optimization strategies
- **Evaluation** - Comprehensive framework with built-in metrics, custom evaluation functions, error handling, batch processing, and detailed result analysis

## Signatures

### Basic Structure

```ruby
class TaskSignature < DSPy::Signature
  description "Clear description of what this signature accomplishes"

  input do
    const :field_name, String
  end

  output do
    const :result_field, String
  end
end
```

### Input Types

```ruby
input do
  const :text, String                       # Required string
  const :context, T.nilable(String)         # Optional string
  const :max_length, Integer                # Required integer
  const :include_score, T::Boolean          # Boolean
  const :tags, T::Array[String]             # Array of strings
  const :metadata, T::Hash[String, String]  # Hash
end
```

### Output Types with Enums

```ruby
class Priority < T::Enum
  enums do
    Low = new('low')
    Medium = new('medium')
    High = new('high')
  end
end

output do
  const :priority, Priority
  const :confidence, Float
end
```

### Default Values (v0.7.0+)

```ruby
class SmartSearch < DSPy::Signature
  description "Search with intelligent defaults"

  input do
    const :query, String
    const :max_results, Integer, default: 10
    const :language, String, default: "English"
  end

  output do
    const :results, T::Array[String]
    const :cached, T::Boolean, default: false
  end
end
```

### Working with Structs

```ruby
class ContactInfo < T::Struct
  const :name, String
  const :email, String
  const :phone, T.nilable(String)
end

class ExtractContact < DSPy::Signature
  description "Extract contact information"

  output do
    const :contact, ContactInfo
  end
end
```

### Union Types (v0.11.0+)

```ruby
# Single-field unions - automatic type detection
class TaskAction < DSPy::Signature
  output do
    const :action, T.any(CreateTask, UpdateTask, DeleteTask)
  end
end

# DSPy automatically adds a _type discriminator field to distinguish
# between different struct types in unions
```

#### Type Discrimination with `_type` Fields

DSPy.rb uses sophisticated type discrimination to handle complex data structures reliably:

```ruby
# Example structs for demonstration
class SearchAction < T::Struct
  const :query, String
  const :max_results, Integer, default: 10
end

class AnswerAction < T::Struct
  const :content, String
  const :confidence, Float
end

# Union type signature
class ActionSignature < DSPy::Signature
  output do
    const :action, T.any(SearchAction, AnswerAction)
  end
end
```

**How `_type` Fields Work:**

1. **Automatic Injection**: DSPy adds `_type` fields to JSON schemas with `const` constraints
2. **Type Resolution**: LLMs include the `_type` field in responses to indicate struct type
3. **Automatic Filtering**: DSPy filters out `_type` fields during deserialization for all structs
4. **Recursive Handling**: Works at any nesting level in complex data structures

**JSON Response Example:**
```json
{
  "action": {
    "_type": "SearchAction",
    "query": "Ruby programming",
    "max_results": 5
  }
}
```

**Important Considerations:**

- **Reserved Field**: Never define your own `_type` fields in T::Struct classes
- **Automatic Filtering**: `_type` is automatically removed during struct creation
- **Union vs Direct**: Both union types and direct struct fields handle `_type` filtering
- **Error Prevention**: Prevents type mismatch errors during deserialization

## Modules

### Creating Custom Modules

```ruby
class SentimentAnalyzer < DSPy::Module
  def initialize
    super
    @predictor = DSPy::Predict.new(SentimentSignature)
  end

  def forward(text:)
    @predictor.call(text: text)
  end
end
```

### Module Composition

```ruby
class DocumentProcessor < DSPy::Module
  def initialize
    super
    @classifier = DocumentClassifier.new
    @summarizer = DocumentSummarizer.new
    @extractor = KeywordExtractor.new
  end

  def forward(document:)
    classification = @classifier.call(content: document)
    summary = @summarizer.call(content: document)
    keywords = @extractor.call(content: document)

    {
      document_type: classification.document_type,
      summary: summary.summary,
      keywords: keywords.keywords
    }
  end
end
```

### Lifecycle Callbacks

Modules expose Rails-style lifecycle hooks so you can instrument cross-cutting concerns without cluttering `forward`.

- `before` callbacks run ahead of `forward` for setup (timers, context loading)
- `around` callbacks wrap `forward` and must `yield`, letting you bracket execution
- `after` callbacks fire once `forward` returns for cleanup, logging, or persistence

Callbacks execute in the order: all `before` hooks → `around` (pre-yield) → `forward` → `around` (post-yield) → `after` hooks. Multiple callbacks of the same type run in registration order.

```ruby
class InstrumentedModule < DSPy::Module
  before :start_timer
  around :with_context
  after :record_metrics

  def initialize
    super
    @predictor = DSPy::Predict.new(QuestionSignature)
  end

  def forward(question:)
    @predictor.call(question: question)
  end

  private

  def start_timer
    @started_at = Time.now
  end

  def with_context
    load_context
    result = yield
    save_context(result)
    result
  end

  def record_metrics
    DSPy.logger.info(duration: Time.now - @started_at)
  end
end
```

### Per-Instance LM Configuration

```ruby
module = DSPy::ChainOfThought.new(SignatureClass)
module.configure do |config|
  config.lm = DSPy::LM.new('anthropic/claude-3-sonnet',
    api_key: ENV['ANTHROPIC_API_KEY']
  )
end
```

## Predictors

### Predict

Basic LLM calls with signatures:

```ruby
predictor = DSPy::Predict.new(EmailClassifier)
result = predictor.call(
  email_content: "My order hasn't arrived",
  sender: "customer@example.com"
)
```

### ChainOfThought

Adds automatic reasoning to any signature:

```ruby
# Automatically adds :reasoning field to output
cot = DSPy::ChainOfThought.new(ComplexAnalysis)
result = cot.call(data: complex_data)
puts result.reasoning  # Step-by-step explanation
```

### ReAct

Tool-using agent with reasoning:

```ruby
# Define tools (you would implement CalculatorTool)
calculator = YourCalculatorTool.new
memory_tools = DSPy::Tools::MemoryToolset.to_tools

# Create agent
agent = DSPy::ReAct.new(
  ResearchSignature,
  tools: [calculator, *memory_tools],
  max_iterations: 10
)

result = agent.call(query: "Calculate compound interest...")
```

### CodeAct

CodeAct now ships in the `dspy-code_act` gem. See [`lib/dspy/code_act/README.md`](https://github.com/vicentereig/dspy.rb/blob/main/lib/dspy/code_act/README.md) for examples, safety recommendations, and advanced usage patterns.

## Rich Types

### Enums

```ruby
class Status < T::Enum
  enums do
    Active = new('active')
    Inactive = new('inactive')
    Pending = new('pending')
  end
end
```

### Structs

```ruby
class Product < T::Struct
  const :name, String
  const :price, Float
  const :tags, T::Array[String], default: []
end
```

### Arrays of Structs

```ruby
output do
  const :products, T::Array[Product]
end

# Automatic conversion from JSON
result.products.each do |product|
  puts "#{product.name}: $#{product.price}"
end
```

### Union Types

```ruby
# Automatic type detection (v0.11.0+)
output do
  const :result, T.any(SuccessResult, ErrorResult)
end

# Pattern matching
case result.result
when SuccessResult
  puts "Success: #{result.result.message}"
when ErrorResult
  puts "Error: #{result.result.error}"
end
```

### Nested Structures

```ruby
class Company < T::Struct
  class Department < T::Struct
    const :name, String
    const :head, String
  end

  const :name, String
  const :departments, T::Array[Department]
end
```

### Recursive Types with `$defs` (v0.34.0+)

DSPy.rb supports recursive types in structured outputs using JSON Schema `$defs`:

```ruby
class TreeNode < T::Struct
  const :value, String
  const :children, T::Array[TreeNode], default: []  # Self-reference
end

class DocumentAST < DSPy::Signature
  description 'Parse document into tree structure'

  output do
    const :root, TreeNode
  end
end
```

The schema generator automatically creates `#/$defs/TreeNode` references for recursive types, compatible with OpenAI and Gemini structured outputs.

**Important**: Use `default: []` instead of `T.nilable(T::Array[...])` for OpenAI compatibility:

```ruby
# ✅ Good - works with OpenAI structured outputs
const :children, T::Array[TreeNode], default: []

# ❌ Bad - causes schema issues with OpenAI
const :children, T.nilable(T::Array[TreeNode])
```

### Field Descriptions for T::Struct (v0.34.0+)

DSPy.rb extends T::Struct to support field-level `description:` kwargs that flow to JSON Schema:

```ruby
class ASTNode < T::Struct
  const :node_type, NodeType, description: 'The type of node (heading, paragraph, etc.)'
  const :text, String, default: "", description: 'Text content of the node'
  const :level, Integer, default: 0  # No description - self-explanatory
  const :children, T::Array[ASTNode], default: []
end

# Access descriptions programmatically
ASTNode.field_descriptions[:node_type]  # => "The type of node (heading, paragraph, etc.)"
```

The generated JSON Schema includes these descriptions, helping LLMs understand field semantics.

**When to use field descriptions**:
- Complex field semantics not obvious from the type
- Enum-like strings with specific allowed values
- Fields with constraints (e.g., "1-6 for heading levels")

**When to skip descriptions**:
- Self-explanatory fields like `name`, `id`, `url`
- Fields where the type tells the story

## Multimodal Support

DSPy.rb provides comprehensive support for text and image inputs through its unified `DSPy::Image` interface, enabling vision-capable LLM applications across multiple providers.

### Image Input Types

```ruby
# URL-based images (OpenAI only)
image = DSPy::Image.new(url: "https://example.com/image.jpg")

# Base64 encoded images (both providers)
image = DSPy::Image.new(
  base64: base64_string,
  content_type: "image/jpeg"
)

# Byte array images (both providers)
File.open("image.jpg", "rb") do |file|
  image = DSPy::Image.new(
    data: file.read,
    content_type: "image/jpeg"
  )
end

# With detail level (OpenAI only)
image = DSPy::Image.new(
  url: "https://example.com/image.jpg",
  detail: "high"
)
```

### Using Images with Raw Chat

Currently, multimodal support works at the raw chat level using the message builder:

```ruby
# Configure with vision-capable model
lm = DSPy::LM.new('openai/gpt-4o-mini', api_key: ENV['OPENAI_API_KEY'])

# Single image analysis
image = DSPy::Image.new(url: "https://example.com/photo.jpg")
response = lm.raw_chat do |messages|
  messages.user_with_image('What is in this image?', image)
end

puts response # String response

# Multiple images
image1 = DSPy::Image.new(url: "https://example.com/before.jpg")
image2 = DSPy::Image.new(url: "https://example.com/after.jpg")

response = lm.raw_chat do |messages|
  messages.user_with_images('Compare these images', [image1, image2])
end

# With system prompt
response = lm.raw_chat do |messages|
  messages.system('You are an expert image analyst.')
  messages.user_with_image('Analyze this image in detail.', image)
end
```

### Supported Formats and Limits

- **Formats**: JPEG, PNG, GIF, WebP
- **Size Limit**: 5MB per image
- **OpenAI**: URLs and base64, supports `detail` parameter
- **Anthropic**: Base64 and raw data only, no `detail` parameter
- **Signatures**: Image fields in signatures are not yet supported; use raw chat for vision.

## Agent Systems

### ReAct Agent

Reasoning + Acting pattern:

```ruby
class ResearchAssistant < DSPy::Module
  def initialize
    super

    # Create tools (implement as needed for your use case)
    calculator = YourCalculatorTool.new
    memory_tools = DSPy::Tools::MemoryToolset.to_tools

    @agent = DSPy::ReAct.new(
      ResearchSignature,
      tools: [calculator, *memory_tools]
    )
  end

  def forward(query:)
    @agent.call(query: query)
  end
end
```

### CodeAct Agent

Install `dspy-code_act` to build Think-Code-Observe agents. The gem ships with a full agent walkthrough in its README.

### Custom Agents

Build your own agent patterns:

```ruby
class CustomAgent < DSPy::Module
  def initialize
    super
    @planner = DSPy::ChainOfThought.new(PlanningSignature)
    # Requires the dspy-code_act gem for Think-Code-Observe execution
    @executor = DSPy::CodeAct.new(ExecutionSignature)
    @validator = DSPy::Predict.new(ValidationSignature)
  end

  def forward(task:)
    plan = @planner.call(task: task)
    execution = @executor.call(plan: plan.plan)
    validation = @validator.call(result: execution.solution)

    {
      result: execution.solution,
      confidence: validation.confidence
    }
  end
end
```

## Memory Systems

### Basic Memory Operations

```ruby
# Initialize memory
DSPy::Memory.configure do |config|
  config.storage_adapter = :in_memory  # or :redis
end

# Store memory
memory_id = DSPy::Memory.manager.store_memory(
  "User prefers dark mode",
  user_id: "user123",
  tags: ["preferences", "ui"]
)

# Retrieve memory
memory = DSPy::Memory.manager.retrieve_memory(memory_id)

# Search memories
memories = DSPy::Memory.manager.search_memories(
  user_id: "user123",
  tags: ["preferences"]
)
```

### Memory with Agents

```ruby
class PersonalAssistant < DSPy::Module
  def initialize
    super
    memory_tools = DSPy::Tools::MemoryToolset.to_tools

    @agent = DSPy::ReAct.new(
      AssistantSignature,
      tools: memory_tools
    )
  end

  def forward(user_message:, user_id:)
    @agent.call(
      user_message: user_message,
      user_id: user_id
    )
  end
end
```

### Redis Storage

```ruby
require 'redis'

DSPy::Memory.configure do |config|
  config.storage_adapter = :redis
  config.redis_client = Redis.new(url: ENV['REDIS_URL'])
  config.redis_namespace = 'dspy:memory'
end
```

## Toolsets

### Creating Tools with Advanced Sorbet Types

Tools now support comprehensive Sorbet type system including enums, structs, arrays, and hashes with automatic JSON conversion:

```ruby
# Enum-based tool with comprehensive type support
class CalculatorTool < DSPy::Tools::Base
  tool_name 'calculator'
  tool_description 'Performs arithmetic operations with type-safe enum inputs'

  class Operation < T::Enum
    enums do
      Add = new('add')
      Subtract = new('subtract')
      Multiply = new('multiply')
      Divide = new('divide')
    end
  end

  sig { params(operation: Operation, num1: Float, num2: Float).returns(T.any(Float, String)) }
  def call(operation:, num1:, num2:)
    case operation
    when Operation::Add then num1 + num2
    when Operation::Subtract then num1 - num2
    when Operation::Multiply then num1 * num2
    when Operation::Divide
      return "Error: Division by zero" if num2 == 0
      num1 / num2
    end
  end
end
```

### Creating Toolsets

Toolsets group related tools together with comprehensive type support:

```ruby
class WeatherToolset < DSPy::Tools::Toolset
  toolset_name "weather"

  class WeatherCondition < T::Enum
    enums do
      Sunny = new('sunny')
      Cloudy = new('cloudy')
      Rainy = new('rainy')
      Snowy = new('snowy')
    end
  end

  class Temperature < T::Struct
    const :celsius, Float
    const :fahrenheit, Float
    const :feels_like, Float
  end

  class WeatherReport < T::Struct
    const :location, String
    const :condition, WeatherCondition
    const :temperature, Temperature
    const :humidity, Integer
    const :wind_speed, Float
    const :timestamp, String
  end

  tool :get_current, tool_name: "weather_current", description: "Get current weather conditions"
  tool :get_forecast, description: "Get detailed weather forecast"

  sig { params(location: String).returns(WeatherReport) }
  def get_current(location:)
    # Actual implementation would call weather API
    WeatherReport.new(
      location: location,
      condition: WeatherCondition::Sunny,
      temperature: Temperature.new(
        celsius: 22.0,
        fahrenheit: 71.6,
        feels_like: 24.0
      ),
      humidity: 60,
      wind_speed: 10.5,
      timestamp: Time.now.iso8601
    )
  end
end

# Convert to tool instances for agents
weather_tools = WeatherToolset.to_tools
```

### Built-in Toolsets

```ruby
# Memory toolset with persistent storage
memory_tools = DSPy::Tools::MemoryToolset.to_tools
# Includes: memory_store, memory_retrieve, memory_search,
#          memory_list, memory_update, memory_delete,
#          memory_clear, memory_count, memory_get_metadata
```

### Automatic Type Conversion

DSPy.rb provides seamless automatic conversion from JSON parameters to Ruby types in tools:

```ruby
# When agents call tools, JSON strings are automatically converted
# Agent provides: { "operation": "add", "num1": 10, "num2": 20 }
# DSPy converts:
# - "add" string → CalculatorTool::Operation::Add enum
# - 10, 20 numbers → Float values
# - Result: tool.call(operation: Add, num1: 10.0, num2: 20.0)

# Conversion works for all Sorbet types:
# - T::Enum → Automatic string-to-enum conversion
# - T::Struct → Recursive hash-to-struct conversion
# - T::Array[Type] → Array element conversion
# - T::Hash[String, Type] → Hash value conversion
# - T.nilable(Type) → Handles null/nil values
# - T.any(Type1, Type2) → Union type resolution
# - Nested combinations → Deep conversion at any level
```

## Optimization

### MIPROv2 Optimization

Advanced multi-prompt optimization with bootstrap sampling and Bayesian optimization:

```ruby
# Auto-configuration modes for different needs
light_optimizer = DSPy::Teleprompt::MIPROv2::AutoMode.light(metric: your_metric)
medium_optimizer = DSPy::Teleprompt::MIPROv2::AutoMode.medium(metric: your_metric)
heavy_optimizer = DSPy::Teleprompt::MIPROv2::AutoMode.heavy(metric: your_metric)

# Custom configuration using dry-configurable pattern
custom_optimizer = DSPy::Teleprompt::MIPROv2.new(metric: custom_metric)
custom_optimizer.configure do |config|
  config.num_trials = 15
  config.num_instruction_candidates = 6
  config.max_bootstrapped_examples = 5
  config.max_labeled_examples = 20
  config.bootstrap_sets = 6
  config.optimization_strategy = :bayesian  # or :greedy, :adaptive
  config.early_stopping_patience = 4
end

# Run optimization
program = DSPy::Predict.new(YourSignature)
result = custom_optimizer.compile(program, trainset: training_examples, valset: validation_examples)

puts "Best MIPROv2 score: #{result.best_score_value}"
puts "Optimized program: #{result.optimized_program}"
puts "Optimization history: #{result.history.length} trials"
```

### GEPA Optimization

Genetic-Pareto reflective prompt evolution that replays traces, collects feedback, and asks a reflection LM to rewrite instructions:

> Install the `dspy-gepa` gem (and set `DSPY_WITH_GEPA=1` when developing inside the monorepo) to load `DSPy::Teleprompt::GEPA`.

```ruby
feedback_map = {
  'self' => ->(predictor_output:, predictor_inputs:, module_inputs:, module_outputs:, captured_trace:) do
    DSPy::Prediction.new(
      score: predictor_output[:answer] == module_outputs[:answer] ? 1.0 : 0.2,
      feedback: "Tie feedback to the original question: #{module_inputs.input_values[:question]}"
    )
  end
}

optimizer = DSPy::Teleprompt::GEPA.new(
  metric: metric,
  feedback_map: feedback_map,
  experiment_tracker: GEPA::Logging::ExperimentTracker.new
)

program = DSPy::Predict.new(YourSignature)
result = optimizer.compile(program, trainset: train_examples, valset: val_examples)

puts "GEPA trials: #{result.history.count}"
puts "Latest reflection: #{result.history.last&.reflection}"
```

### Evaluation Framework

DSPy.rb provides a comprehensive evaluation system for testing and measuring LLM application performance.

#### Basic Evaluation

```ruby
# Create evaluator with a predictor and metric
predictor = DSPy::Predict.new(YourSignature)
metric = DSPy::Metrics.exact_match(field: :answer)
evaluator = DSPy::Evals.new(predictor, metric: metric)

# Evaluate against test examples
result = evaluator.evaluate(test_examples, display_progress: true)

puts "Score: #{result.score}"
puts "Passed: #{result.passed_examples}/#{result.total_examples}"
```

#### Built-in Metrics

```ruby
# Exact match comparison
exact_metric = DSPy::Metrics.exact_match(field: :answer, case_sensitive: false)

# Contains/substring matching
contains_metric = DSPy::Metrics.contains(field: :answer)

# Numeric difference with tolerance
numeric_metric = DSPy::Metrics.numeric_difference(field: :score, tolerance: 0.1)

# Composite AND (all must pass)
composite_metric = DSPy::Metrics.composite_and(exact_metric, contains_metric)
```

#### Custom Metrics

```ruby
# Custom metric as proc
custom_metric = ->(example, prediction) do
  return false unless prediction && prediction.respond_to?(:answer)
  prediction.answer.downcase.strip == example.expected_answer.downcase.strip
end

# Multi-factor custom metric
quality_metric = ->(example, prediction) do
  return 0.0 unless prediction

  score = 0.0
  score += 0.5 if prediction.answer == example.expected_answer  # Accuracy
  score += 0.3 if prediction.explanation&.length&.> 50          # Completeness
  score += 0.2 if prediction.confidence&.> 0.8                  # Confidence

  score
end

evaluator = DSPy::Evals.new(predictor, metric: quality_metric)
```

## Production Features

### Observability

**Configuration Requirements:**

Add the optional gems:
```ruby
gem 'dspy'
gem 'dspy-o11y'
gem 'dspy-o11y-langfuse'
```

Langfuse integration also requires these environment variables:
```bash
export LANGFUSE_PUBLIC_KEY=pk_your_public_key
export LANGFUSE_SECRET_KEY=sk_your_secret_key
# Optional: defaults to https://cloud.langfuse.com
export LANGFUSE_HOST=https://your-langfuse-instance.com
```

**Basic Setup:**

```ruby
# Enable structured logging
DSPy.configure do |c|
  c.logger = Dry.Logger(:dspy, formatter: :json)
end

# Observability automatically activates when Langfuse env vars are present
# Span tracking emitted for:
# - llm.generate
# - dspy.predict
# - module.forward
# - tool.execute
```

### Error Handling

```ruby
begin
  result = predictor.call(input: data)
rescue DSPy::Errors::ValidationError => e
  puts "Invalid input: #{e.message}"
rescue DSPy::Errors::LMError => e
  puts "LLM error: #{e.message}"
  # Implement retry logic
end
```

### Event System

DSPy.rb includes a comprehensive event system for monitoring and extending functionality:

**Dual Event APIs:**

```ruby
# Simple logging (legacy)
DSPy.log('custom.event', data: 'value', context: 'info')

# Structured event system (recommended)
DSPy.event('custom.event', { data: 'value', context: 'info' })
```

**Event Subscription:**

```ruby
# Subscribe to specific events
subscription_id = DSPy.events.subscribe('llm.*') do |event_name, attributes|
  puts "LLM Event: #{event_name}"
  puts "Attributes: #{attributes.inspect}"
end

# Subscribe to all events
all_events_id = DSPy.events.subscribe('*') do |event_name, attributes|
  # Custom processing for all events
  MyCustomLogger.log(event_name, attributes)
end

# Unsubscribe when done
DSPy.events.unsubscribe(subscription_id)
```

**Built-in Events:**

- `llm.generate` - LLM API calls with token usage and timing
- `dspy.predict` - Prediction operations with inputs/outputs
- `module.forward` - Module execution with span tracking
- `tool.execute` - Tool invocations in agents
- `span.start` / `span.end` - OpenTelemetry span lifecycle

## Testing Strategies

### Unit Testing with RSpec

```ruby
RSpec.describe EmailClassifier do
  let(:classifier) { EmailClassifier.new }

  it "classifies spam correctly" do
    result = classifier.call(
      email_content: "Win a prize!",
      sender: "spam@example.com"
    )

    expect(result.category).to eq("spam")
    expect(result.confidence).to be > 0.8
  end
end
```

### Integration Testing with VCR

```ruby
RSpec.describe "LLM Integration", vcr: true do
  let(:predictor) { DSPy::Predict.new(AnalysisSignature) }

  it "analyzes text with real LLM" do
    result = predictor.call(text: "Sample text")
    expect(result).to respond_to(:analysis)
  end
end
```

### Mocking LLM Responses

```ruby
# In tests
allow(predictor).to receive(:call).and_return(
  DSPy::Prediction.new(
    sentiment: "positive",
    confidence: 0.9
  )
)
```

## API Reference

### Core Classes

#### DSPy::Signature
- `description(text)` - Set signature description
- `input { }` - Define input schema
- `output { }` - Define output schema
- `.input_json_schema` - Get input JSON schema
- `.output_json_schema` - Get output JSON schema

#### DSPy::Module
- `initialize` - Constructor
- `forward(**kwargs)` - Main processing method
- `call(**kwargs)` - Alias for forward
- `configure { |config| }` - Configure module

#### DSPy::Prediction
- Automatic type conversion from JSON
- Access fields as methods
- Handles enums, structs, arrays

### Predictors

#### DSPy::Predict
- `new(signature_class)` - Create predictor
- `call(**inputs)` - Execute prediction

#### DSPy::ChainOfThought
- Adds `:reasoning` field automatically
- Same API as Predict

#### DSPy::ReAct
- `new(signature, tools:, max_iterations: 10)`
- Returns result with tool call history

#### DSPy::CodeAct
- `new(signature, max_iterations: 8)` (requires the `dspy-code_act` gem)
- Returns solution and execution history

### Configuration

#### DSPy.configure
```ruby
DSPy.configure do |c|
  c.lm                              # Language model
  c.logger                          # Structured logger
  c.strategy                        # Extraction strategy
end
```

## Integration Guides

### Rails Integration

```ruby
# config/initializers/dspy.rb
Rails.application.config.after_initialize do
  DSPy.configure do |c|
    c.lm = DSPy::LM.new(
      Rails.application.credentials.llm_model,
      api_key: Rails.application.credentials.llm_api_key
    )
    c.logger = Dry.Logger(:dspy, formatter: Rails.env.production? ? :json : :string)
  end
end

# app/services/email_classifier_service.rb
class EmailClassifierService
  def initialize
    @classifier = DSPy::ChainOfThought.new(EmailClassifier)
  end

  def classify(email)
    @classifier.call(
      email_content: email.body,
      sender: email.from
    )
  end
end
```

### Sidekiq Jobs

```ruby
class AnalyzeDocumentJob
  include Sidekiq::Job

  def perform(document_id)
    document = Document.find(document_id)

    analyzer = DSPy::Predict.new(DocumentAnalysis)
    result = analyzer.call(content: document.text)

    document.update!(
      category: result.category,
      summary: result.summary
    )
  end
end
```

## Examples

Additional runnable examples in the repo:
- Workflow router: `examples/workflow_router.rb`
- Evaluator + optimizer loop: `examples/evaluator_loop.rb`
- GitHub assistant agent: `examples/github-assistant/`

### Email Support System

```ruby
# Signature for email classification
class EmailTriage < DSPy::Signature
  description "Triage customer support emails"

  class Priority < T::Enum
    enums do
      Low = new('low')
      Medium = new('medium')
      High = new('high')
      Urgent = new('urgent')
    end
  end

  input do
    const :subject, String
    const :body, String
    const :customer_tier, String
  end

  output do
    const :department, String
    const :priority, Priority
    const :summary, String
    const :auto_reply_suggested, T::Boolean
  end
end

# Agent with memory
class SupportAgent < DSPy::Module
  def initialize
    super

    memory_tools = DSPy::Tools::MemoryToolset.to_tools

    @triage = DSPy::ChainOfThought.new(EmailTriage)
    @agent = DSPy::ReAct.new(
      SupportResponse,
      tools: memory_tools
    )
  end

  def forward(email:, customer_id:)
    # Triage email
    triage_result = @triage.call(
      subject: email.subject,
      body: email.body,
      customer_tier: email.customer.tier
    )

    # Generate response with context
    response = @agent.call(
      email: email.body,
      customer_id: customer_id,
      priority: triage_result.priority.serialize
    )

    {
      department: triage_result.department,
      priority: triage_result.priority,
      response: response.suggested_reply,
      should_escalate: triage_result.priority == EmailTriage::Priority::Urgent
    }
  end
end
```

## Best Practices

1. **Signature Design**
   - Clear, specific descriptions
   - Appropriate type constraints
   - Meaningful field names

2. **Module Composition**
   - Single responsibility principle
   - Dependency injection
   - Testable components

3. **Error Handling**
   - Graceful degradation
   - Retry strategies
   - User-friendly messages

4. **Production Deployment**
   - Enable monitoring
   - Set up alerts
   - Version your modules

## Resources

- **Documentation**: https://oss.vicente.services/dspy.rb/
- **GitHub**: https://github.com/vicentereig/dspy.rb
- **Issues**: https://github.com/vicentereig/dspy.rb/issues
- **Examples**: https://github.com/vicentereig/dspy.rb/tree/main/examples

## Version History

- v0.34.0 - Recursive schema `$defs` fix, T::Struct field descriptions
- v0.33.0 - Previous release
- See CHANGELOG.md for full history

---

Generated for DSPy.rb v0.34.0
