#include <stdio.h>
#define LEARNING_RATE 0.01
#define EPOCHS 1000
#define DATA_SIZE 4

// Function to calculate the forward pass
double forward_pass(double x1, double x2, double w1, double w2) {
    return w1 * x1 + w2 * x2;
}

// Function to calculate the mean squared error (MSE) loss
double mean_squared_error(double y_true[], double y_pred[]) {
    double sum = 0.0;
    for (int i = 0; i < DATA_SIZE; ++i) {
        double error = y_true[i] - y_pred[i];
        sum += error * error;
    }
    return sum / DATA_SIZE;
}

// Function to calculate the gradient of the loss with respect to weights
void gradient(double x1[], double x2[], double y_true[], double w1, double w2, double *dw1, double *dw2) {
    *dw1 = 0.0;
    *dw2 = 0.0;

    for (int i = 0; i < DATA_SIZE; ++i) {
        double y_pred = forward_pass(x1[i], x2[i], w1, w2);
        double error = y_pred - y_true[i];

        *dw1 += 2 * error * x1[i];
        *dw2 += 2 * error * x2[i];
    }

    *dw1 /= DATA_SIZE;
    *dw2 /= DATA_SIZE;
}

int main() {
    double x1[DATA_SIZE] = {1, 2, 0, -2};
    double x2[DATA_SIZE] = {0, 1, 1, 1};
    double y_true[DATA_SIZE] = {1, 9, 1, 7};

    // Initial weights
    double w1 = 0.5;
    double w2 = -0.5;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        double dw1, dw2;

        // Calculate gradients
        gradient(x1, x2, y_true, w1, w2, &dw1, &dw2);

        // Update weights using gradient descent
        w1 -= LEARNING_RATE * dw1;
        w2 -= LEARNING_RATE * dw2;

        // Calculate and print the loss
        double y_pred[DATA_SIZE];
        for (int i = 0; i < DATA_SIZE; ++i) {
            y_pred[i] = forward_pass(x1[i], x2[i], w1, w2);
        }
        double loss = mean_squared_error(y_true, y_pred);

        if (epoch % 100 == 0) {
            printf("Epoch %d, Loss: %lf\n", epoch, loss);
        }
    }

    // Print the final weights
    printf("Final weights: w1 = %lf, w2 = %lf\n", w1, w2);

    return 0;
}
