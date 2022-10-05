from node import from_string, draw_tree

lt = """
(TOP (NP (NNP Federal) (NNP Home) (NNP Loan) (NNP Mortgage) (NNP Corp.)))
s"""
t = from_string(lt)
draw_tree(t, 'ct2')


lp = """
(TOP (NP (_ Federal) (_ Home) (_ Loan) (_ Mortgage) (_ Corp) (_ .)))
""".replace('_', '\<pos\>')
t = from_string(lp)
draw_tree(t, 'cp2')

# from supar import Parser
# parser = Parser.load('crf-con-en')
# dataset = parser.predict('I saw Sarah with a telescope.', lang='en', prob=True, verbose=True)
# print(dataset.sentences)
