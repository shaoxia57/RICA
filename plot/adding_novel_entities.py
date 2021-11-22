from pandas import DataFrame
import matplotlib.pyplot as plt
   
Data = {'# of novel entities': [100, 300, 500,600,700, 1000,1500, 2000],
        'Accuracy': [47.21, 48.60, 49.99, 52.38, 91.66, 97.91, 100, 100]
       }
x_axis = [100, 300, 500,600,700, 1000,1500, 2000]
y_axis = [47.21, 48.60, 49.99, 52.38, 91.66, 97.91, 100, 100]
x_plot = range(len(y_axis))
# df = DataFrame(Data,columns=['# of novel entities','Accuracy'])

# df.plot(x ='# of novel entities', y='Accuracy', kind = 'line', label='roberta-base')
plt.plot(x_plot, y_axis)
plt.xticks(x_plot, x_axis)
plt.xlabel('# of novel entities')
plt.ylabel('Accuracy(%)')
# plt.ylim(0,100)

plt.show()

