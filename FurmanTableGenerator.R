library(magrittr)
library(kable)
library(kableExtra)


data = read.csv("C:/Users/u009192/Desktop/Furman_parameters_compare.csv")
names(data) <- c('n', '\\hat{\\bm \\alpha}_{Miles}', '\\hat{\\bm \\theta}_{Miles}', '\\hat{\\bm \\alpha}', '\\hat{\\bm \\theta}')
data = data[data$n <= 5,]

rez_table = data %>% 
  dplyr::mutate_all(round,4) %>% 
  dplyr::mutate_if(is.numeric,format,digits=4) %>%
  knitr::kable(escape=FALSE,format = "latex", booktabs=TRUE, label = "table:furman_compare", caption = "Shapes and scales from the projection of a log-normal with Miles's algorithm compared to our projection.") %>%
  kableExtra::kable_styling(full_width = T, latex_options = c("HOLD_position")) %>%
  kableExtra::pack_rows("n=2",1,2) %>%
  kableExtra::pack_rows("n=3",3,5) %>%
  kableExtra::pack_rows("n=4",6,9) %>%
  kableExtra::pack_rows("n=5",10,14) %>% 
  kableExtra::add_header_above(c(" " = 1, "Miles" = 2, "Laguerre" = 2))


fileConn<-file("furman_compare.tex")
writeLines(rez_table, fileConn)
close(fileConn)
