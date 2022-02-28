# Setting environment
# install.packages("tidyverse")
# install.packages("ploty")

library(tidyverse)
library(plotly)

# setwd("D:/CHU 2.0/Forest/110-1 Data Mining/Final project")

train = read.csv("train.csv", row.names = "id")
test = read.csv("test.csv", row.names = "id")
all = rbind(train, test) %>% # Combine training and testing data
  select(-Gender, -Customer.Type, -Type.of.Travel, -Class)

# Interactive plot used to check distribution of features
# Satisfied  # Change x for different feature
p_satisfied = ggplot(all %>% filter(satisfaction == "satisfied"), aes(x = Departure.Delay.in.Minutes)) + 
  geom_histogram(fill = "pink") +
  ggtitle("Departure Delay in Minutes_neutral or dissatisfied") + 
  xlab(NULL) +
  ylab("Count") +
  theme_minimal() + 
  theme(plot.title = element_text(hjust = 0.5))
ggplotly(p_satisfied)

# Dissatisfied  # Change x for different feature
p_dissatisfied = ggplot(all %>% filter(satisfaction == "neutral or dissatisfied"), aes(x = Departure.Delay.in.Minutes)) + 
  geom_histogram(fill = "grey") +
  ggtitle("Departure Delay in Minutes_neutral or dissatisfied") + 
  xlab(NULL) +
  ylab("Count") +
  theme_minimal() + 
  theme(plot.title = element_text(hjust = 0.5))
ggplotly(p_dissatisfied)