🧠 AI Agent Framework

  Welcome to the AI Agent Framework, a comprehensive, modular, and extensible system for building your own intelligent agents.
  This project leverages modern AI technologies including Large Language Models (LLMs), tool augmentation, memory systems, and reasoning engines
	 
🚀 Features

	🌐 Natural Language Understanding (NLU)

	🧠 Short-term & Long-term Memory (with vector search)

	💡 LLM-powered Reasoning Engine (OpenAI-compatible)

	🛠️ Tool Use Framework (Web Search, Weather, and more)

	🔁 Continuous learning loop (perception → reasoning → action → feedback)

	🧪 Full testing suite with unit, integration, and behavioral tests

	🌐 Web & CLI interfaces

Configure Environment Variables
Create a .env file:

	OPENAI_API_KEY=your_openai_key
	SEARCH_API_KEY=your_search_key
	WEATHER_API_KEY=your_weather_key

💻 Usage
Command Line Interface

	python src/main.py

Web Interface

	python src/web_app.py

Open your browser at http://localhost:5000


🧪 Testing
Run all tests:

	pytest tests/

Includes:

✅ Unit tests (perception, memory, tools)

🔄 Integration tests (end-to-end conversation flow)

🤖 Behavioral tests (safety, fact-checking, etc.)


🐳 Docker Support
To build and run the agent in a Docker container:

	docker build -t ai-agent .
	docker run -p 8000:8000 --env-file .env ai-agent

🛠️ Extending

* Add tools: src/tools/tool_interface.py

* Customize reasoning: src/reasoning/reasoning_engine.py

* Modify memory: src/memory/

* Create new actions: src/action/action_generator.py


📄 License

This project is released under the MIT License.