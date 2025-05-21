# 🧠 LLM-Agent+: A Modular Framework for Intelligent Agents

Welcome to LLM-Agent+, a comprehensive, modular, and extensible framework for building intelligent agents powered by Large Language Models (LLMs). This project integrates natural language understanding, a dual-layer memory system, a reasoning engine with chain-of-thought capabilities, and a standardized interface for external tool integration.

## 🚀 Key Features

- 🌐 **Natural Language Understanding (NLU)** - Parse user inputs and extract intents and entities
- 🧠 **Dual-layer Memory System**
  - Short-term memory (STM) for recent conversational history
  - Long-term memory (LTM) with vector-embedded, persistent storage (FAISS/Pinecone)
- 💡 **Advanced Reasoning Engine**
  - Chain-of-Thought (CoT) prompting for complex problem-solving
  - Self-Refinement strategies for iterative improvement
  - Plan-and-solve workflows for structured reasoning
- 📝 **Reasoning Trace Compression (RTC)**
  - Dynamic compression of reasoning traces to reduce token usage
  - Segmentation of traces into logical blocks
  - Salience scoring to identify important segments
  - Summarization of low-importance segments
  - Consistency validation to ensure logical coherence
- 🛠️ **Tool Integration Layer**
  - Standardized tool schema for easy extension
  - OpenAPI/Swagger support for automatic API wrapping
  - Semantic router for matching queries to appropriate tools
- 🔄 **Continuous Learning Loop** (perception → reasoning → action → feedback)
- 🧪 **Comprehensive Testing Suite** with unit, integration, and behavioral tests
- 🌐 **Multiple Interfaces** - Web UI and CLI with visualization capabilities

## 📊 Performance Benefits

- **Token Efficiency**: RTC reduces token usage by 58-72% in empirical evaluations
- **Logical Coherence**: Maintains >95% entailment score on consistency tests
- **Memory Optimization**: Dual-layer memory system enables context retention across extended interactions

## 🔧 Configuration

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_key
SEARCH_API_KEY=your_search_key
WEATHER_API_KEY=your_weather_key
```

## 💻 Usage

### Command Line Interface

```bash
python src/main.py
```

### Web Interface

```bash
python src/web_app.py
```

Open your browser at [http://localhost:5000](http://localhost:5000)

The web interface features:
- Real-time reasoning trace visualization
- Memory exploration via nearest-neighbor search
- Interactive debugging tools

## 🧪 Testing

Run all tests:

```bash
pytest tests/
```

Includes:
- ✅ Unit tests (perception, memory, tools, RTC)
- 🔄 Integration tests (end-to-end conversation flow)
- 🤖 Behavioral tests (safety, fact-checking, etc.)

## 🐳 Docker Support

To build and run the agent in a Docker container:

```bash
docker build -t llm-agent-plus .
docker run -p 8000:8000 --env-file .env llm-agent-plus
```

## 🛠️ Extending the Framework

LLM-Agent+ is designed for high modularity and extensibility:

### Adding New Tools
- Implement the tool interface in `src/tools/tool_interface.py`
- Register your tool with the tool manager
- Tools are automatically available to the reasoning engine

### Customizing the Reasoning Engine
- Modify prompts and strategies in `src/reasoning/reasoning_engine.py`
- Adjust RTC parameters in `src/reasoning/rtc/trace_compressor.py`
- Create custom reasoning workflows

### Enhancing Memory Systems
- Extend vector store capabilities in `src/memory/`
- Implement custom retrieval strategies
- Add new memory backends (e.g., Pinecone, Weaviate)

### Creating New Actions
- Add action handlers in `src/action/action_generator.py`
- Implement domain-specific response formatting

## 🔬 System Architecture

LLM-Agent+ follows a modular architecture with these core components:

1. **Natural Language Understanding (NLU)** - Parses user inputs into structured representations
2. **Memory System** - Manages short-term and long-term memory
3. **Reasoning Engine** - Drives problem-solving with Chain-of-Thought and RTC
4. **Tool Integration Layer** - Connects to external APIs and services
5. **Action Generation** - Formulates responses and executes commands

The typical execution flow:
1. User submits input via CLI or web UI
2. NLU extracts structured meaning
3. Memory modules retrieve relevant context
4. Reasoning engine constructs a CoT reasoning chain
5. RTC compresses the reasoning trace when needed
6. Tools are invoked through the Tool Integration Layer
7. Action Generator returns the final response
8. Memory is updated with the new experience

## 📄 License

This project is released under the MIT License.
