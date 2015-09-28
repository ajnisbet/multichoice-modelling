# Clear console
rm(list = ls())
cat("\014")
set.seed(1)


# Load data
df <- read.csv('df.csv', header=TRUE)
df$is_correct <- as.logical(df$is_correct)
df$is_first <- as.logical(df$is_first)
df$is_float <- as.logical(df$is_float)
df$is_fraction <- as.logical(df$is_fraction)
df$is_integer <- as.logical(df$is_integer)
df$is_last <- as.logical(df$is_last)
df$is_none_of_the_above <- as.logical(df$is_none_of_the_above)
df$is_numeric <- as.logical(df$is_numeric)
df$is_percent <- as.logical(df$is_percent)
df$is_unit <- as.logical(df$is_unit)
print(summary(df))

# Split into testing and training
training.size <- round(0.75 * nrow(df))
training.indices <- sample(nrow(df), training.size)
training.data <- df[training.indices, ]
testing.data <- df[-training.indices, ]


# Fit model
lm <- glm(is_correct ~ is_first, data=training.data, family="binomial")

# Obtain prediction
predictions <- rep(NA, nrow(testing.data))
probs <- predict(lm, testing.data, type="response")
predictions[probs > 0.5] <- "Yes"
predictions[probs < 0.5] <- "No"
print(mean(predictions != testing.data$default))