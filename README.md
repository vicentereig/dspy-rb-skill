# DSPy.rb Claude Skill

[![DSPy.rb](https://img.shields.io/badge/DSPy.rb-v0.34.1-red)](https://github.com/vicentereig/dspy.rb)
[![Claude Skill](https://img.shields.io/badge/Claude-Skill-blueviolet)](https://support.claude.com/en/articles/12512198-how-to-create-custom-skills)

A [Claude Skill](https://support.claude.com/en/articles/12512180-using-skills-in-claude) that helps you build type-safe LLM applications using [DSPy.rb](https://github.com/vicentereig/dspy.rb).

## What This Skill Does

When activated, Claude gains deep knowledge of DSPy.rb to help you:

- **Define type-safe signatures** with Sorbet types (enums, structs, arrays)
- **Build composable modules** for complex LLM workflows
- **Create ReAct and CodeAct agents** with tool calling
- **Use recursive types** with proper `$defs` JSON Schema format
- **Add field descriptions** to T::Struct for better LLM understanding
- **Implement optimization** using MIPROv2 and GEPA
- **Test and evaluate** LLM applications with RSpec and VCR
- **Deploy to production** with observability and error handling

## Installation

### Claude Code

Clone this repository into your global skills directory:

```bash
git clone https://github.com/vicentereig/dspy-rb-skill ~/.claude/skills/dspy-rb
```

Or add it to a specific project:

```bash
git clone https://github.com/vicentereig/dspy-rb-skill .claude/skills/dspy-rb
```

### Claude.ai (Pro/Max/Team/Enterprise)

1. Download this repository as a ZIP file
2. Go to Claude.ai Settings > Skills
3. Upload the ZIP file

## Usage

Once installed, Claude will automatically activate this skill when you:

- Ask about building LLM applications in Ruby
- Mention DSPy.rb, signatures, or predictors
- Need help with type-safe AI patterns
- Work with ReAct or CodeAct agents

### Example Prompts

- "Help me create a DSPy signature for classifying customer emails"
- "How do I use ChainOfThought for multi-step reasoning?"
- "Create a ReAct agent with custom tools"
- "Set up MIPROv2 optimization for my predictor"

## Skill Contents

| File | Description |
|------|-------------|
| `SKILL.md` | Core skill instructions (~200 lines) |
| `REFERENCE.md` | Complete API reference (~2400 lines) |

## Resources

- **Documentation**: https://oss.vicente.services/dspy.rb/
- **GitHub**: https://github.com/vicentereig/dspy.rb
- **RubyGems**: https://rubygems.org/gems/dspy

## License

MIT License - Same as [DSPy.rb](https://github.com/vicentereig/dspy.rb/blob/main/LICENSE)
