---
name: dspy-rb
description: Build type-safe LLM applications with DSPy.rb - Ruby's programmatic prompt framework with signatures, modules, agents, and optimization
---

# DSPy.rb

> Build LLM apps like you build software. Type-safe, modular, testable.

DSPy.rb brings software engineering best practices to LLM development. Instead of tweaking prompts, you define what you want with Ruby types and let DSPy handle the rest.

## Overview

DSPy.rb is a Ruby framework for building language model applications with programmatic prompts. It provides:

- **Type-safe signatures** - Define inputs/outputs with Sorbet types
- **Modular components** - Compose and reuse LLM logic
- **Automatic optimization** - Use data to improve prompts, not guesswork
- **Production-ready** - Built-in observability, testing, and error handling

## Core Concepts

### 1. Signatures
Define interfaces between your app and LLMs using Ruby types:

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
    const :priority, Priority  # Type-safe enum with defined values
    const :confidence, Float
  end
end
```

### 2. Modules
Build complex workflows from simple building blocks:

- **Predict** - Basic LLM calls with signatures
- **ChainOfThought** - Step-by-step reasoning
- **ReAct** - Tool-using agents
- **CodeAct** - Dynamic code generation agents (install the `dspy-code_act` gem)

#### Lifecycle callbacks
Rails-style lifecycle hooks ship with every `DSPy::Module`, so you can wrap `forward` without touching instrumentation:

- **`before`** – runs ahead of `forward` for setup (metrics, context loading)
- **`around`** – wraps `forward`, calls `yield`, and lets you pair setup/teardown logic
- **`after`** – fires after `forward` returns for cleanup or persistence

### 3. Tools & Toolsets
Create type-safe tools for agents with comprehensive Sorbet support:

```ruby
# Enum-based tool with automatic type conversion
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

# Multi-tool toolset with rich types
class DataToolset < DSPy::Tools::Toolset
  toolset_name "data_processing"

  class Format < T::Enum
    enums do
      JSON = new('json')
      CSV = new('csv')
      XML = new('xml')
    end
  end

  class ProcessingConfig < T::Struct
    const :max_rows, Integer, default: 1000
    const :include_headers, T::Boolean, default: true
    const :encoding, String, default: 'utf-8'
  end

  tool :convert, description: "Convert data between formats"
  tool :validate, description: "Validate data structure"

  sig { params(data: String, from: Format, to: Format, config: T.nilable(ProcessingConfig)).returns(String) }
  def convert(data:, from:, to:, config: nil)
    config ||= ProcessingConfig.new
    "Converted from #{from.serialize} to #{to.serialize} with config: #{config.inspect}"
  end

  sig { params(data: String, format: Format).returns(T::Hash[String, T.any(String, Integer, T::Boolean)]) }
  def validate(data:, format:)
    {
      valid: true,
      format: format.serialize,
      row_count: 42,
      message: "Data validation passed"
    }
  end
end
```

### 4. Type System & Discriminators
DSPy.rb uses sophisticated type discrimination for complex data structures:

- **Automatic `_type` field injection** - DSPy adds discriminator fields to structs for type safety
- **Union type support** - T.any() types automatically disambiguated by `_type`
- **Reserved field name** - Avoid defining your own `_type` fields in structs
- **Recursive filtering** - `_type` fields filtered during deserialization at all nesting levels

### 5. Optimization
Improve accuracy with real data:

- **MIPROv2** - Advanced multi-prompt optimization with bootstrap sampling and Bayesian optimization
- **GEPA (Genetic-Pareto Reflective Prompt Evolution)** - Reflection-driven instruction rewrite loop with feedback maps, experiment tracking, and telemetry
- **Evaluation** - Comprehensive framework with built-in and custom metrics, error handling, and batch processing

## Quick Start

```ruby
# Install
gem 'dspy'

# Configure
DSPy.configure do |c|
  c.lm = DSPy::LM.new('openai/gpt-4o-mini', api_key: ENV['OPENAI_API_KEY'])
  # or use Ollama for local models
  # c.lm = DSPy::LM.new('ollama/llama3.2')
end

# Define a task
class SentimentAnalysis < DSPy::Signature
  description "Analyze sentiment of text"

  input do
    const :text, String
  end

  output do
    const :sentiment, String  # positive, negative, neutral
    const :score, Float       # 0.0 to 1.0
  end
end

# Use it
analyzer = DSPy::Predict.new(SentimentAnalysis)
result = analyzer.call(text: "This product is amazing!")
puts result.sentiment  # => "positive"
puts result.score      # => 0.92
```

## Provider Adapter Gems

Add the adapter gems that match the providers you call:

```ruby
# Gemfile
gem 'dspy'
gem 'dspy-openai'    # OpenAI, OpenRouter, Ollama
gem 'dspy-anthropic' # Claude
gem 'dspy-gemini'    # Gemini
```

Each adapter gem already pulls in the official SDK (`openai`, `anthropic`, `gemini-ai`), so you don't need to add those manually.

## Key URLs

- Homepage: https://oss.vicente.services/dspy.rb/
- GitHub: https://github.com/vicentereig/dspy.rb
- Documentation: https://oss.vicente.services/dspy.rb/getting-started/

## Guidelines for Claude

When helping users with DSPy.rb:

1. **Focus on signatures** - They define the contract with LLMs
2. **Use proper types** - T::Enum for categories, T::Struct for complex data
3. **Leverage automatic type conversion** - Tools and toolsets automatically convert JSON strings to proper Ruby types (enums, structs, arrays, hashes)
4. **Compose modules** - Chain predictors for complex workflows
5. **Create type-safe tools** - Use Sorbet signatures for comprehensive tool parameter validation and conversion
6. **Test thoroughly** - Use RSpec and VCR for reliable tests
7. **Monitor production** - Enable Langfuse by installing the optional o11y gems and setting env vars

## See Also

For complete API reference, advanced patterns, and integration guides, see [REFERENCE.md](REFERENCE.md).

## Version

Current: 0.33.0
