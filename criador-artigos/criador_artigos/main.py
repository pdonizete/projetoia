from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from crewai_tools import SerperDevTool, ScrapeWebsiteTool  # , DallETool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import logging

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Configuração do logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.FileHandler("detalhamento_execucao.log")
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

# Testando o logger manualmente
logger.debug("Logger configurado corretamente e pronto para uso.")

# LLMs

# Gemini
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError(
        "A chave de API do Google não foi fornecida. Verifique o arquivo .env."
    )
gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0.5,
    google_api_key=google_api_key,
)

# Llama
grog_api_key = os.getenv("GROQ_API_KEY")
if not grog_api_key:
    raise ValueError(
        "A chave de API do GROG não foi fornecida. Verifique o arquivo .env."
    )
llama_3 = ChatGroq(
    api_key=grog_api_key,
    model="llama3-70b-8192",  # "llama-3.1-70b-versatile"
    timeout=180,
)

# ChatGPT
gpt_4o = ChatOpenAI(model_name="gpt-4o")
gpt_4o_mini = ChatOpenAI(model_name="gpt-4o-mini")

# Ollama
"""
os.environ["OPENAI_API_KEY"] = "NA"
ollama = ChatOllama(
    model="llama3.1",
    base_url="http://localhost:11434",
)"""
ollama = ChatOpenAI(
    model="llama3.1",
    base_url="http://localhost:11434",
)


# Definindo LLM e rpm maximo padrao
DEFAULT_LLM = gpt_4o_mini
DEFAULT_MAX_RPM = 30

# Ferramenta de pesquisa na internet
search_tool = SerperDevTool()
search_tool.country = "BR"
search_tool.location = "São Paulo"
search_tool.locale = "pt-BR"
search_tool.n_results = 25
scrape_tool = ScrapeWebsiteTool()

# Ferramenta DALL-E com especificação de tamanho 16:9
# dalle_tool = DallETool(size="1024x1024")  # Formato 16:9

# 1. Pesquisador de Design de Jogos
pesquisador_temas = Agent(
    role="Pesquisador de Temas",
    goal="Realizar pesquisas sobre o {tema}.",
    backstory=(
        "você é um pesquisador experiente e apaixonado em buscar coisas novas sobre qualquer tema"
    ),
    verbose=True,
    memory=True,
    tools=[search_tool],  # Utiliza ferramentas de pesquisa e LLM
    llm=DEFAULT_LLM,  # Liga o LLM diretamente ao agente
    allow_delegation=False,
)

tarefa_pesquisa_tema = Task(
    description=(
        "Pesquise informações detalhadas e abrangentes sobre o {tema}. "
        "foque nos aspectos dos dias de hoje, no impacto que o {tema} terá para quem está lendo"
    ),
    expected_output="o resultado das pesquisas realizadas",

    tools=[search_tool],  # Ferramentas usadas para a pesquisa
    agent=pesquisador_temas,

    logger=logger,
)

# 2. Pesquisador de Interface de Usuário (UI) para Jogos de Terminal
redator = Agent(
    role="redator",
    goal=(
"redigir um artigo baseado na pesquisa do 'Pesquisador de Temas'"
"o artigo deve ser escrito de forma a atrair potenciais leitores"
"ele deve está pronto para ser publicado em blogs e redes sociais"
"ele deve ser didático e fácil de entender."
),
    backstory=(
"você é especialista em redigir artigos"
"seu trabalho é a produção de artigos que atraiam o leitor e o faça se apaixonar pelo {tema}"
    ),
    verbose=True,
    memory=True,
    tools=[search_tool],  # Utiliza ferramentas de pesquisa e LLM
    llm=gpt_4o,  # Liga o LLM diretamente ao agente
    allow_delegation=False,
)

tarefa_redator = Task(
    description=(
        "Com base na pesquisa realizada pelo agente 'Pesquisador de Temas' sobre o tema {tema}, "
        "desenvolva um artigo detalhado que dê ao usuário um amplo conhecimento sobre o {tema}"
    ),
    tools=[search_tool],  # Ferramenta usada para gerar a proposta baseada na pesquisa
    agent=redator,
        expected_output="o artigo escrito sobre o {tema}",

    logger=logger,
)

# 6. Agente Revisor
revisor = Agent(
    role = "Revisor de Conteúdo",
    goal="Revisar todo o conteúdo produzido e entregar a versão final para o usuário",
    verbose=True,
    memory=True,
    backstory=(
        "Você tem um olho afiado para detalhes, garantindo que todo o conteúdo "
        "esteja perfeito antes de ser entregue ao usuário."
        "você será capaz de verificar se o artigo mostra evidências de que foi produzido por uma ia"
        "caso isso aconteça, você será capaz de corrigir de forma a evitar esse problema"
    ),
    llm=llama_3,
    allow_delegation=False,
)

tarefa_revisor = Task(
    description=(
        "Revisar todo o conteúdo produzido e entregar a versão final para o usuário."
        "Todo o texto deve estar em Português Brasil."
    ),
    expected_output="Conteúdo revisado, pronto para entrega ao usuário.",
    agent=revisor,
    output_file="artigo_final.md",  # Configurando o output para salvar em um arquivo Markdown
)

# Formando a crew
crew = Crew(
    agents=[
        pesquisador_temas,
        redator,
        revisor,
    ],
    tasks=[
        tarefa_pesquisa_tema,
        tarefa_redator,
        tarefa_revisor,

    ],
    verbose=True,
    logger=logger,
    manager_llm=DEFAULT_LLM,
    function_calling_llm=DEFAULT_LLM,
    max_rpm=DEFAULT_MAX_RPM,
    max_iter=3,

    process=Process.sequential,  # Processamento sequencial das tarefas
    allow_delegation=False,
)

tema = input("Digite o tema: ")

print(tema)

# Inicia a execução da tarefa
result = crew.kickoff(inputs={"tema": tema})  # Exemplo com o jogo UNO
logger.info(result)
print("Execução detalhada salva em detalhamento_execucao.log")
