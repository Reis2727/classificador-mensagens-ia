# Classificador de Mensagens com Inteligência Artificial
Projeto acadêmico do Projeto Integrador, voltado à aplicação de aprendizado de máquina na classificação de mensagens em português.
## Objetivo
Explorar como a Inteligência Artificial pode auxiliar na organização de mensagens, sugerindo uma das categorias:
- Trabalho
- Estudos
- Pessoal
## Como funciona
O programa utiliza 24 mensagens fictícias, sendo oito de cada categoria, como exemplos de treinamento.
O TF-IDF transforma os textos em representações numéricas. Em seguida, o algoritmo Multinomial Naive Bayes aprende padrões associados às categorias e sugere uma classificação para a mensagem digitada.
Quando o texto não contém nenhuma palavra reconhecida pelo modelo, o programa informa que não há informação suficiente para classificar.
## Tecnologias
- Python
- scikit-learn
- TF-IDF
- Multinomial Naive Bayes
- GitHub para armazenamento e controle de versões
## Arquivos
- `classificador.py`: código do programa e exemplos de treinamento.
- `requirements.txt`: biblioteca necessária.
- `.gitignore`: configuração para evitar o versionamento de arquivos auxiliares.
- `README.md`: apresentação e instruções.
## Como executar
É necessário ter Python e pip instalados.
1. Baixe o repositório pelo botão Code → Download ZIP.
2. Extraia o arquivo ZIP.
3. Abra um terminal na pasta extraída.
4. Instale a dependência:
```bash
python -m pip install -r requirements.txt
```
5. Execute:
```bash
python classificador.py
```
6. Digite uma mensagem para receber a categoria sugerida.
7. Digite `sair` para encerrar.
## Limitações
Este é um protótipo educativo com uma base pequena de exemplos. Pode errar, especialmente em mensagens ambíguas ou diferentes das usadas no treinamento.
O modelo não aprende automaticamente com as mensagens digitadas. Também não foi realizada uma avaliação com uma base independente de testes, portanto não há uma taxa de acerto comprovada.
## Próximas melhorias
- Ampliar e diversificar os exemplos.
- Separar dados de treinamento e teste.
- Avaliar os resultados com métricas de classificação.
- Criar uma interface gráfica.
