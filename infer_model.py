import pandas as pd
from make_pred_single import extract_entity
# from data import data



# if we have an input file, use this line.
test =pd.read_csv('input/test.csv')
print(test.head)


# test_sen = data[400:430]
# test_sen = test_sen.reset_index(drop=True)
# test = test_sen['text']

print("Predicting output result")
result = extract_entity(test['text'])

print(result)
result.to_csv('output/output.csv', index=False)