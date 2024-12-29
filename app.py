from flask import Flask, render_template, request
import pandas as pd
import math
import numpy as np
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Retrieve form data
        predict_x1 = pd.to_numeric(request.form['p1[0]'], errors='coerce')
        
        predict_x2 = pd.to_numeric(request.form['p2[1]'], errors='coerce')

        data = pd.DataFrame({
            'x1': pd.to_numeric(request.form.getlist('x1[]'), errors='coerce'),
            'x2': pd.to_numeric(request.form.getlist('x2[]'), errors='coerce'),
            'y':pd.to_numeric(request.form.getlist('y2[]'), errors='coerce')
        })
        # Training data
       
        
        # Compute statistics for Gaussian Naïve Bayes
        stats = compute_gnb_stats(data)

        # Step-by-step details
        detailed_steps = ""
        class_probs = {}
        for cls, class_stats in stats.items():
            # Compute probabilities for x1 and x2
            prob_x1 = gaussian_pdf(predict_x1, class_stats['x1_mean'], class_stats['x1_std'])
            prob_x2 = gaussian_pdf(predict_x2, class_stats['x2_mean'], class_stats['x2_std'])
            prior = class_stats['prior']
            
            # Calculate total probability
            class_probs[cls] = prob_x1 * prob_x2 * prior
            
            # Add details for each step
            detailed_steps += f"<h4>Class {cls}</h4>"
            detailed_steps += f"<p>Step 1: Prior probability (P(Class)) = {prior:.4f}</p>"
            detailed_steps += f"<p>Step 2: Compute Gaussian probabilities for features:</p>"
            detailed_steps += f"<ul><li>P(x1|Class) = (1 / sqrt(2πσ²)) * exp(-((x1 - μ)² / 2σ²))</li>"
            detailed_steps += f"<li>P(x1|Class) = (1 / sqrt(2π * {class_stats['x1_std']:.4f}²)) * exp(-(({predict_x1} - {class_stats['x1_mean']:.4f})² / 2 * {class_stats['x1_std']:.4f}²))</li>"
            detailed_steps += f"<li>P(x1|Class) = {prob_x1:.6f}</li>"
            detailed_steps += f"<li>P(x2|Class) = {prob_x2:.6f}</li></ul>"
            detailed_steps += f"<p>Step 3: Multiply probabilities with prior:</p>"
            detailed_steps += f"<p>P(Class|x) = P(x1|Class) * P(x2|Class) * P(Class)</p>"
            detailed_steps += f"<p>P(Class|x) = {prob_x1:.6f} * {prob_x2:.6f} * {prior:.4f} = {class_probs[cls]:.8f}</p>"

        # Determine the classification result
        classification = max(class_probs, key=class_probs.get)
        detailed_steps += f"<h4>Final Step: Compare probabilities</h4>"
        detailed_steps += f"<p>Class 0: {class_probs[0]:.8f}</p>"
        detailed_steps += f"<p>Class 1: {class_probs[1]:.8f}</p>"
        detailed_steps += f"<p>Result: Predicted Class = {classification}</p>"

        # Generate outputs
        table_html = generate_table_html(data, stats)
        equations_latex = generate_gnb_equations(stats, predict_x1, predict_x2)
        return render_template('solver.html', 
                               table_latex=table_html,
                               equations_latex=equations_latex,
                               steps_latex=detailed_steps,
                               result_latex=f"Classification: {classification}")

    return render_template('index.html')


def compute_gnb_stats(data):
    """Compute statistics for Gaussian Naïve Bayes."""
    stats = {}
    for cls in data['y'].unique():
        subset = data[data['y'] == cls]
        stats[cls] = {
            'x1_mean': subset['x1'].mean(),
            'x1_std': subset['x1'].std(),
            'x2_mean': subset['x2'].mean(),
            'x2_std': subset['x2'].std(),
            'prior': len(subset) / len(data)
        }
    return stats

def gaussian_pdf(x, mean, std):
    """Calculate Gaussian probability density function."""
    exponent = math.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

def generate_table_html(data, stats):
    """Generate LaTeX table for the dataset and computed statistics."""
    
    # Generate LaTeX table for dataset (x1, x2, Class)
    latex_table = """
    \\begin{array}{|c|c|c|} \\hline
    x_1 & x_2 & Class \\\\ \\hline
    """
    
    for _, row in data.iterrows():
        latex_table += f"{row['x1']} & {row['x2']} & {row['y']} \\\\ \\hline\n"
    
    latex_table += """
    \\end{array}
    \n\n
    """
    
    # Generate LaTeX table for statistics (Class, Mean, Std, Prior)
    latex_stats_table = """
    \\begin{array}{|c|c|c|c|c|c|} \\hline
    Class & x_1 \\text{ Mean} & x_1 \\text{ Std} & x_2 \\text{ Mean} & x_2 \\text{ Std} & Prior \\\\ \\hline
    """
    
    for cls, stat in stats.items():
        latex_stats_table += f"{cls} & {stat['x1_mean']:.2f} & {stat['x1_std']:.2f} & {stat['x2_mean']:.2f} & {stat['x2_std']:.2f} & {stat['prior']:.2f} \\\\ \\hline\n"
    
    latex_stats_table += """
    \\end{array}
    """
    
    return latex_table + latex_stats_table


def generate_gnb_equations(stats, x1, x2):
    """Generate LaTeX equations dynamically for Gaussian Naïve Bayes."""
    equations = """
    $$ P(Class) = \\frac{\\text{Count of Class}}{\\text{Total Samples}} $$
    $$ P(X|Class) = \\frac{1}{\\sqrt{2\\pi \\sigma^2}} e^{-\\frac{(X - \\mu)^2}{2\\sigma^2}} $$
    $$ \\mu = \\frac{\\sum X}{n}, \\quad \\sigma = \\sqrt{\\frac{\\sum (X - \\mu)^2}{n}} $$
    """
    for cls, class_stats in stats.items():
        equations += f"""
        $$ \\text{{Class {cls}: }} $$
        $$ \\mu_{{x1}} = \\frac{{\\sum x1}}{{n}} = {class_stats['x1_mean']:.2f}, \\quad \\mu_{{x2}} = \\frac{{\\sum x2}}{{n}} = {class_stats['x2_mean']:.2f} $$
        $$ \\sigma_{{x1}} = \\sqrt{{\\frac{{\\sum (x1 - \\mu_{{x1}})^2}}{{n}}}} = {class_stats['x1_std']:.2f}, \\quad \\sigma_{{x2}} = \\sqrt{{\\frac{{\\sum (x2 - \\mu_{{x2}})^2}}{{n}}}} = {class_stats['x2_std']:.2f} $$
        $$ P(x1|Class) = \\frac{{1}}{{\\sqrt{{2\\pi \\left( {class_stats['x1_std']:.2f}^2 \\right)}}}} e^{{-\\frac{{({x1} - {class_stats['x1_mean']:.2f})^2}}{{2 \\left( {class_stats['x1_std']:.2f}^2 \\right)}}}} $$
        $$ P(x2|Class) = \\frac{{1}}{{\\sqrt{{2\\pi \\left( {class_stats['x2_std']:.2f}^2 \\right)}}}} e^{{-\\frac{{({x2} - {class_stats['x2_mean']:.2f})^2}}{{2 \\left( {class_stats['x2_std']:.2f}^2 \\right)}}}} $$
        """
    return equations

if __name__ == '__main__':
    app.run(debug=True)