import matplotlib.pyplot as plt
import textwrap


def custom_autopct(pct):
    return ('%.2f%%' % pct) if pct > 0.1 else ''


def create_pie_chart(data, title, colors=None, title_size=16, label_size=14, legend_size=14, wrap_width=60):

    wrapped_title = "\n".join(textwrap.wrap(title, wrap_width))
    labels = [f"{key} ({value})" for key, value in data.items()]
    values = list(data.values())

    # Create the figure with more width to accommodate the legend
    plt.figure(figsize=(11, 8))
    wedges, texts, autotexts = plt.pie(
        values,
        labels=None,  # Hide labels on the pie slices to avoid overlap
        autopct=custom_autopct,  # Display percentage with one decimal place
        startangle=90,  # Start the first slice at 90 degrees
        colors=colors,  # Apply the custom colors if provided
        pctdistance=0.7  # Default distance for most labels
    )

    # Set the font size for the percentage labels on the pie chart
    for autotext in autotexts:
        autotext.set_fontsize(label_size)

    # Manually adjust the position of a specific label, e.g., the second label
    # Adjusting only the label for "Agree" (index 1 in this example)
    autotexts[4].set_position((1.5 * autotexts[4].get_position()[0],
                               1.5 * autotexts[4].get_position()[1]))

    # Adjust the title and place it at the top
    plt.title(wrapped_title, fontsize=title_size, y=1.05)

    # Adjust the legend to be on the right side without overlapping the pie chart
    plt.legend(
        wedges, labels, fontsize=legend_size, title_fontsize=legend_size,
        loc="center left", bbox_to_anchor=(1, 0.5), borderaxespad=0
    )

    # Ensure everything fits within the figure size
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for the legend
    plt.show()


# Example usage:
data_example = {
    "Strongly Agree": 100,
    "Agree": 48,
    "Basically Agree": 50,
    "Disagree": 4,
    "Strongly Disagree": 2
}

# Define custom colors for blue shades
custom_colors = ['#cce5ff', '#99ccff', '#66b2ff', '#3399ff', '#0073e6']

# Call the function with the long title
create_pie_chart(
    data_example,
    "Question 39: The park provides a good security environment for visitors and surrounding communities",
    colors=custom_colors,
    title_size=16,  # Larger title font size
    label_size=14,  # Adjusted percentage label size
    legend_size=14  # Adjusted legend font size
)
