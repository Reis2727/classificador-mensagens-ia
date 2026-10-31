"""Classificador educativo de mensagens em português.
Usa exemplos fictícios e aprendizado de máquina.
A base é pequena: as classificações podem conter erros.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

EXEMPLOS = {
   "Trabalho": [
       "A reunião da equipe será amanhã no escritório",
       "Preciso entregar o relatório ao gerente",
       "O cliente solicitou uma proposta comercial",
       "Vamos organizar as tarefas do projeto",
       "Envie o contrato para o departamento responsável",
       "O prazo para finalizar o serviço termina hoje",
       "Precisamos revisar o orçamento da empresa",
       "O funcionário recebeu uma promoção",
   ],
   "Estudos": [
       "Preciso estudar para a prova da faculdade",
       "O professor passou uma atividade para a turma",
       "Vou revisar o conteúdo da disciplina",
       "A aula de programação começa amanhã",
       "Precisamos entregar o trabalho acadêmico",
       "Estou lendo um livro para a pesquisa",
       "O curso tem exercícios de matemática",
       "Minha matrícula na universidade foi confirmada",
   ],
   "Pessoal": [
       "Vamos jantar com a família amanhã",
       "Hoje é o aniversário do meu amigo",
       "Preciso comprar alimentos no supermercado",
       "Vamos passear no parque no domingo",
       "Estou planejando uma viagem de férias",
       "Vou assistir a um filme em casa",
       "Meu irmão convidou todos para um churrasco",
       "Vou levar meu cachorro para passear",
   ],
}

def criar_modelo():
   """Transforma os textos em números e treina o classificador."""
   textos = []
   categorias = []
   for categoria, mensagens in EXEMPLOS.items():
       textos.extend(mensagens)
       categorias.extend([categoria] * len(mensagens))
   modelo = make_pipeline(
       TfidfVectorizer(strip_accents="unicode"),
       MultinomialNB(),
   )
   modelo.fit(textos, categorias)
   return modelo

def classificar(modelo, mensagem):
   """Evita classificar textos sem palavras conhecidas pelo modelo."""
   vetor = modelo.steps[0][1].transform([mensagem])
   if vetor.nnz == 0:
       return "Sem informação suficiente para classificar"
   return modelo.predict([mensagem])[0]

def main():
   modelo = criar_modelo()
   print("CLASSIFICADOR DE MENSAGENS COM IA")
   print("Categorias: Trabalho, Estudos e Pessoal.")
   print("Protótipo educativo: pode cometer erros.")
   print("Digite 'sair' para encerrar.\n")
   while True:
       try:
           mensagem = input("Digite uma mensagem: ").strip()
       except (EOFError, KeyboardInterrupt):
           print("\nPrograma encerrado.")
           break
       if mensagem.casefold() == "sair":
           print("Programa encerrado.")
           break
       if not mensagem:
           print("Digite uma mensagem antes de continuar.\n")
           continue
       resultado = classificar(modelo, mensagem)
       print(f"Categoria sugerida: {resultado}\n")

if __name__ == "__main__":
   main()
