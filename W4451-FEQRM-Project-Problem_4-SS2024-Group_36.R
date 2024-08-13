################################################################################
#                                                                              #
#           W4451 - Applied project using R, summer semester 2024              #
#                                                                              #
#                  Problem 4: GARCH Models and Backtesting                     #
#                                                                              #
################################################################################

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # Automatic adjustment of the working directory to the the directory of this script file; you may need to install the R-package "rstudioapi" first.

# Information:

# Use this file to save your code for solving Problem 4 of the applied project.

# <--------------- Begin in the next line with your own code  ---------------> #

library(readr)     
library(dplyr)     
library(ggplot2) 

data <- read_csv("MSFT.csv")
head(data)

# date range
first_date <- min(data$Date)
last_date <- max(data$Date)

# Print the date range
cat("The data ranges from", first_date, "to", last_date, "\n")

# Plot the closing prices
ggplot(data, aes(x = Date, y = Close)) +
  geom_line() +
  labs(title = "Daily Closing Prices",
       x = "Date",
       y = "Closing Price (USD)") +
  theme_minimal()







# log-returns
data <- data %>%
  mutate(Log_Returns = log(Close / lag(Close))) %>%
  na.omit() 

# Plot the log-returns
ggplot(data, aes(x = Date, y = Log_Returns)) +
  geom_line() +
  labs(title = "Log-returns of the closing price",
       x = "Date",
       y = "Log-Returns") +
  theme_minimal()


# Plot the ACF of the log-returns
acf(data$Log_Returns, main = "ACF of Log-Returns", lag.max = 50)

# Plot the ACF of the squared log-returns
acf(data$Log_Returns^2, main = "ACF of Squared Log-Returns", lag.max = 50)











library(ufRisk)
library(rugarch)
library(dplyr)


data <- data %>% mutate(Log_Returns = log(Close / lag(Close))) %>% na.omit()

# for 250 observations
n_test <- 250
train_data <- data$Log_Returns[1:(nrow(data) - n_test)]
test_data <- data$Log_Returns[(nrow(data) - n_test + 1):nrow(data)]



# Define the GARCH(1,1) model
garch_spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0)),
  distribution.model = "norm"
)

# Fit the GARCH(1,1) model
garch_fit <- ugarchfit(spec = garch_spec, data = train_data, out.sample = n_test)

# Forecast the GARCH(1,1) model for the test period
garch_forecast <- ugarchforecast(garch_fit, n.ahead = 1, n.roll = n_test - 1)
garch_forecast_var <- sigma(garch_forecast) * qnorm(0.99)
garch_forecast_es <- sigma(garch_forecast) * dnorm(qnorm(0.975)) / 0.975

# Define the APARCH(1,1) model
aparch_spec <- ugarchspec(
  variance.model = list(model = "apARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0)),
  distribution.model = "norm"
)

# Fit the APARCH(1,1) model
aparch_fit <- ugarchfit(spec = aparch_spec, data = train_data, out.sample = n_test)

# Forecast the APARCH(1,1) model for the test period
aparch_forecast <- ugarchforecast(aparch_fit, n.ahead = 1, n.roll = n_test - 1)
aparch_forecast_var <- sigma(aparch_forecast) * qnorm(0.99)
aparch_forecast_es <- sigma(aparch_forecast) * dnorm(qnorm(0.975)) / 0.975

# traffic light test
traffic_light_test <- function(realized, predicted, alpha) {
  exceptions <- sum(realized < -predicted)
  expected_exceptions <- length(realized) * alpha
  ratio <- exceptions / expected_exceptions
  
  if (ratio < 0.8) {
    result <- "Green light zone: Model is acceptable."
  } else if (ratio < 1.2) {
    result <- "Yellow light zone: Model is on the borderline."
  } else {
    result <- "Red light zone: Model is not acceptable."
  }
  
  list(exceptions = exceptions, ratio = ratio, result = result)
}

# Backtesting using the traffic light test for GARCH(1,1) VaR
garch_test <- traffic_light_test(test_data, garch_forecast_var, 0.01)

# Backtesting using the traffic light test for APARCH(1,1) VaR
aparch_test <- traffic_light_test(test_data, aparch_forecast_var, 0.01)

# Print the results of the traffic light test
print("GARCH(1,1) Traffic Light Test Results:")
print(garch_test)

print("APARCH(1,1) Traffic Light Test Results:")
print(aparch_test)

# Function to backtest ES
backtestES <- function(realized, predicted, alpha) {
  es_exceedances <- realized < -predicted
  num_exceedances <- sum(es_exceedances)
  expected_exceedances <- length(realized) * alpha
  
  list(num_exceedances = num_exceedances, expected_exceedances = expected_exceedances)
}

# Backtesting ES for GARCH and APARCH models
garch_es_test <- backtestES(test_data, garch_forecast_es, 0.025)
aparch_es_test <- backtestES(test_data, aparch_forecast_es, 0.025)

# Print the ES backtesting results
print("GARCH(1,1) ES Backtesting Results:")
print(garch_es_test)

print("APARCH(1,1) ES Backtesting Results:")
print(aparch_es_test)













# Prepare data for GARCH(1,1)
garch_test_data <- data.frame(
  Date = data$Date[(nrow(data) - n_test + 1):nrow(data)],
  Negative_Test_Returns = -test_data,
  VaR_99 = as.numeric(garch_forecast_var),
  ES_97_5 = as.numeric(garch_forecast_es)
)

#  points where negative returns exceed VaR
garch_test_data$Exceeds_VaR <- garch_test_data$Negative_Test_Returns > garch_test_data$VaR_99

# Plot GARCH(1,1) results
ggplot(garch_test_data, aes(x = as.Date(Date))) +
  geom_line(aes(y = Negative_Test_Returns), color = "black", alpha = 0.6) +
  geom_line(aes(y = VaR_99), color = "red", linetype = "dashed") +
  geom_line(aes(y = ES_97_5), color = "green", linetype = "dotted") +
  geom_point(aes(y = Negative_Test_Returns, color = Exceeds_VaR), size = 2) +
  scale_color_manual(values = c("TRUE" = "red", "FALSE" = "blue")) +
  labs(title = "GARCH(1,1) 99%-VaR and 97.5%-ES",
       y = "Value",
       x = "Date") +
  theme_minimal()





# Prepare data for APARCH(1,1)
aparch_test_data <- data.frame(
  Date = data$Date[(nrow(data) - n_test + 1):nrow(data)],
  Negative_Test_Returns = -test_data,
  VaR_99 = as.numeric(aparch_forecast_var),
  ES_97_5 = as.numeric(aparch_forecast_es)
)

# Identify points where negative returns exceed VaR
aparch_test_data$Exceeds_VaR <- aparch_test_data$Negative_Test_Returns > aparch_test_data$VaR_99

# Plot APARCH(1,1) results
ggplot(aparch_test_data, aes(x = as.Date(Date))) +
  geom_line(aes(y = Negative_Test_Returns), color = "black", alpha = 0.6) +
  geom_line(aes(y = VaR_99), color = "red", linetype = "dashed") +
  geom_line(aes(y = ES_97_5), color = "green", linetype = "dotted") +
  geom_point(aes(y = Negative_Test_Returns, color = Exceeds_VaR), size = 2) +
  scale_color_manual(values = c("TRUE" = "red", "FALSE" = "blue")) +
  labs(title = "APARCH(1,1) 99%-VaR and 97.5%-ES",
       y = "Value",
       x = "Date") +
  theme_minimal()












# Fit GARCH(1,1) model
garch_spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0)),
  distribution.model = "norm"
)

garch_fit <- ugarchfit(spec = garch_spec, data = train_data, out.sample = 250)
garch_params <- coef(garch_fit)

# Extract GARCH(1,1) valuies
omega_garch <- garch_params["omega"]
alpha_garch <- garch_params["alpha1"]
beta_garch <- garch_params["beta1"]

# Fit APARCH(1,1) model
aparch_spec <- ugarchspec(
  variance.model = list(model = "apARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0)),
  distribution.model = "norm"
)

aparch_fit <- ugarchfit(spec = aparch_spec, data = train_data, out.sample = 250)
aparch_params <- coef(aparch_fit)

# Extract APARCH(1,1) values
omega_aparch <- aparch_params["omega"]
alpha_aparch <- aparch_params["alpha1"]
beta_aparch <- aparch_params["beta1"]
gamma_aparch <- aparch_params["gamma1"]
delta_aparch <- aparch_params["delta"]

# Print the formulas
cat("GARCH(1,1) Model Formula:\n")
cat(paste0("σ_t^2 = ", round(omega_garch, 6), " + ", round(alpha_garch, 6), " * ε_{t-1}^2 + ", round(beta_garch, 6), " * σ_{t-1}^2\n"))

cat("\nAPARCH(1,1) Model Formula:\n")
cat(paste0("σ_t^", round(delta_aparch, 6), " = ", round(omega_aparch, 6), " + ", round(alpha_aparch, 6), " * (|ε_{t-1}| - ", round(gamma_aparch, 6), " * ε_{t-1})^", round(delta_aparch, 6), " + ", round(beta_aparch, 6), " * σ_{t-1}^", round(delta_aparch, 6), "\n"))

