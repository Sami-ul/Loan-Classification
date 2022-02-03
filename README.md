# Loan-Classification
DATASET USED NOT INCLUDED
- Predict if a customer will or will not default
- Important features
  - Debt to income ratio: The reason this may be an important feature in deciding whether a user will default or not because the more debt and less income a person has, the less chance that they will be able to pay back a loan
  - Delinquent credit lines: This may be an important feature because it shows that the persons payments are way past due, it shows that they are not able to pay back their credit cards and therefore may have a hard time paying back a loan
## Models Created
### Decision Tree Model
- The training data performance for the first decision tree model
![image](https://user-images.githubusercontent.com/72280649/152285333-a31e3079-5cde-4405-b358-3c0216b6d563.png)

- The test data performance for the first decision tree model
![image](https://user-images.githubusercontent.com/72280649/152285357-aa7c0f40-ab2e-44c9-b57d-84f47604c712.png)

### Decision Tree Model Tuned
- The training data performance for the decision tree tuned model
![image](https://user-images.githubusercontent.com/72280649/152285475-191e5d56-caa9-4132-b4d8-d9a7f39ee7c4.png)

- The test data performance for the decision tree tuned model
![image](https://user-images.githubusercontent.com/72280649/152285501-8475b477-8140-4e86-a7db-184044615087.png)

### Random Forest Model
- The training data performance for the random forest model
![image](https://user-images.githubusercontent.com/72280649/152285560-d772e570-0e95-4b5d-8319-96250dcbe692.png)

- The test data performance for the random forest model
![image](https://user-images.githubusercontent.com/72280649/152285590-4a5bb49c-f2b6-44c0-b11d-d5dc2598ed4c.png)

### Random Forest Model Tuned
- The training data performance for the final random forest tuned model
![image](https://user-images.githubusercontent.com/72280649/152285627-113dab44-73c4-4b21-9fe2-df7432ef8e2b.png)

- The test data performance for the final random forest tuned model
![image](https://user-images.githubusercontent.com/72280649/152285665-0fd9645a-0fe2-49ea-9dc6-6af0c575cd6d.png)

- As you can see, the untuned models on training data have perfect performance, meaning they are overfit, which leads to worse performance on test data
- To fix this I tuned the data, which equalized the performance and made it so it could work on more varied data
- The models used were decision tree and random forest
- Decision tree was faster but random forest was better at classifying

## Data Insights
- The graph below shows the data imbalance between loans that were paid back, and loans that were defaulted.
- We can see that there is a severe imbalance.
- In the future tuning we may want to fix this.
- ![image](https://user-images.githubusercontent.com/72280649/152285732-e6684919-bbf6-4431-8a54-f95f64566383.png)

- The graph to the right is a density plot of the loan amounts.
- We can see that most of the loans lie around the $20,000 mark.
- This is important because it shows what the risk of getting a prediction wrong is.
- ![image](https://user-images.githubusercontent.com/72280649/152286049-38290dda-b8ce-4db2-afb4-62f46c6921bd.png)

- This graph shows the amount of loans that are for debt consolidation versus the loans that are taken for home improvement.
- This tells us that there are many more people who are getting a loan for debt, rather than for home improvement.
- ![image](https://user-images.githubusercontent.com/72280649/152285801-fb67a267-b5fd-419e-bce4-5135fcd1a371.png)

