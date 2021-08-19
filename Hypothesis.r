library(magrittr) 
library(dplyr) 

errors <- read.csv('output/errors_overall.csv')

summary(errors)

attach(errors)

#Hypothesis 1

# One way ANOVA
# Null :  Weather group SSIM means are not different from the overall mean of the data
# Alternate :  Weather group SSIM means are different from the overall mean of the data

#2x results
res_weather <- errors %>% select(Weather, SSIM, PSNR, Resolution) %>% filter(Resolution == "2x")


group_by(res_weather, Weather) %>%
  summarise(
    count = n(),
    mean = mean(SSIM, na.rm = TRUE),
    sd = sd(SSIM, na.rm = TRUE)
  )

library("ggpubr")
ggboxplot(res_2x_weather, x = "Weather", y = "SSIM", 
          color = "Weather", palette = c("#00AFBB", "#E7B800", "#FC4E07"),
          ylab = "SSIM", xlab = "Weather")

# Compute the analysis of variance
res.aov <- aov(SSIM ~ Weather, data = res_weather)
# Summary of the analysis
summary(res.aov)
TukeyHSD(res.aov)

#4x results
res_weather <- errors %>% select(Weather, SSIM, PSNR, Resolution) %>% filter(Resolution == "4x")


group_by(res_weather, Weather) %>%
  summarise(
    count = n(),
    mean = mean(SSIM, na.rm = TRUE),
    sd = sd(SSIM, na.rm = TRUE)
  )

library("ggpubr")
ggboxplot(res_2x_weather, x = "Weather", y = "SSIM", 
          color = "Weather", palette = c("#00AFBB", "#E7B800", "#FC4E07"),
          ylab = "SSIM", xlab = "Weather")

# Compute the analysis of variance
res.aov <- aov(SSIM ~ Weather, data = res_weather)
# Summary of the analysis
summary(res.aov)
TukeyHSD(res.aov)

#Hypothesis 2

# Z- test
# Null :  Time of the scenes SSIM means are not different from the overall mean 
# Alternate :  Time of the scenes SSIM means are different from the overall mean

#2x results
res_time <- errors %>% select(Time, SSIM, PSNR, Resolution) %>% filter(Resolution == "2x")


group_by(res_time, Time) %>%
  summarise(
    count = n(),
    mean = mean(SSIM, na.rm = TRUE),
    sd = sd(SSIM, na.rm = TRUE)
  )

library("ggpubr")
ggboxplot(res_time, x = "Time", y = "SSIM", 
          color = "Time", palette = c("#00AFBB", "#E7B800", "#FC4E07"),
          ylab = "SSIM", xlab = "Time")

# Compute the analysis of variance
anv_time <- aov(SSIM ~ Time, data = res_time)
# Summary of the analysis
summary(anv_time)
TukeyHSD(res.aov)

#4x results
res_time <- errors %>% select(Time, SSIM, PSNR, Resolution) %>% filter(Resolution == "4x")


group_by(res_time, Time) %>%
  summarise(
    count = n(),
    mean = mean(SSIM, na.rm = TRUE),
    sd = sd(SSIM, na.rm = TRUE)
  )

library("ggpubr")
ggboxplot(res_time, x = "Time", y = "SSIM", 
          color = "Time", palette = c("#00AFBB", "#E7B800", "#FC4E07"),
          ylab = "SSIM", xlab = "Time")

# Compute the analysis of variance
anv_time <- aov(SSIM ~ Time, data = res_time)
# Summary of the analysis
summary(anv_time)
TukeyHSD(res.aov)



# Further analysis 

res_scene <- errors %>% select(Scene, SSIM) %>% filter(Resolution == "2x")

levels(res_2x_scene$Scene)


group_by(res_2x_scene, Scene) %>%
  summarise(
    count = n(),
    mean = mean(SSIM, na.rm = TRUE),
    sd = sd(SSIM, na.rm = TRUE)
  )
 
# Compute the analysis of variance
res.aov <- aov(SSIM ~ Scene, data = res_2x_scene)
# Summary of the analysis
summary(res.aov)
TukeyHSD(res.aov)

install.packages("ggridges")
library(ggridges)
library(ggplot2)
library(dplyr)
library(tidyr)
library(forcats)

# Load dataset from github

errors_om <- read.csv('other_models/om_errors.csv')

m1 <- errors_om %>% filter(errors_om$Scale == "4x")

# Plot
m1 %>%
  mutate(text = fct_reorder(Model, PSNR)) %>%
  ggplot( aes(y=Model, x=PSNR,  fill=Model)) +
  geom_density_ridges(alpha=0.6, stat="binline", bins=40) +
  theme_ridges() +
  theme(
    legend.position="none",
    panel.spacing = unit(0.1, "lines"),
    strip.text.x = element_text(size = 8)
  ) +
  xlab("") +
  ylab("Assigned Probability (%)")


