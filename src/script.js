$(document).ready(function() {
    // Resizing the textarea dynamically
    $('#prediction-input').on('input', function() {
        // Create a span element to measure the text width
        let span = $('<span></span>').css({
            fontSize: $(this).css('font-size'),
            fontFamily: $(this).css('font-family'),
            fontWeight: $(this).css('font-weight'),
            letterSpacing: $(this).css('letter-spacing'),
            whiteSpace: 'pre', // Preserve whitespace
            visibility: 'hidden', // Hide the span
            position: 'absolute', // Position off-screen
        }).appendTo('body'); // Append to the body

        // Set the span text to the current value of the input
        span.text($(this).val() || $(this).attr('placeholder')); // Use placeholder if empty

        // Calculate new width
        const newWidth = span.width() + 20; // Add some padding (20px for left and right)
        
        // Update the input field's width and height
        this.style.height = 'auto'; // Reset height to auto to calculate new height
        this.style.height = (this.scrollHeight) + 'px'; // Set to scrollHeight
        this.style.width = newWidth + 'px'; // Set to new width

        // Remove the span element after measuring
        span.remove();
    });

    // Initialize
    let predictionInput = $(".input-prediction");
    let updatedRotation;

    function updateDisplayMessage(message, fontSize) {
        $(".displayed-prediction").html(message);
        $(".displayed-prediction").css({'font-size': fontSize});
    }

    // Event listener for the 'Predict' button click
    $('.predict-button').click(function() {
        let inputValue = predictionInput.val().trim();
        
        // Check if the input is a valid number
        if (!/^\d*\.?\d+$/.test(inputValue) || inputValue === '') {
            updateDisplayMessage("Invalid input.<br>Please enter a number between 0 and 100.", '1em');
            $(".predictionbox-score").css({"transform": "rotate(-45deg)"});
            $(".needle").css({"transform": "translateX(-50%) translateY(0) rotate(-90deg)", "transform-origin": "bottom"});
            return;
        }

        let predictionValue = parseFloat(inputValue);

        // Input validation
        if (predictionValue < 0) {
            predictionValue = 0;
            updateDisplayMessage("Invalid probability. Rolled back to 0%", '1.5em');
        } else if (predictionValue > 100) {
            predictionValue = 100;
            updateDisplayMessage("Invalid probability. Rolled back to 100%", '1.5em');
        } else {
            updateDisplayMessage("Probability: " + predictionValue.toFixed(2) + "%", '1.5em');
        }

        // Update rotation only if predictionValue is valid
        if (typeof predictionValue === 'number') {
            

            // Send the input to the Flask server if the input is valid
            $.ajax({
                type: "POST",
                url: "/predict",
                contentType: "application/json",
                data: JSON.stringify({ probability: predictionValue }),
                success: function(response) {
                    updatedRotation = Math.round(response.formatted_probability * 180 / 100) - 45;
                    needleRotation = Math.round(response.formatted_probability * 180 / 100) - 90;
                    $(".displayed-prediction").append("<br>Predicted Class: " + response.predicted_class);
                    $(".predictionbox-score").css({"transform": "rotate(" + updatedRotation + "deg)"});
                    $(".needle").css({"transform": "translateX(-50%) translateY(0) rotate(" + needleRotation + "deg)"});
                },
                error: function() {
                    $(".displayed-prediction").append("<br>Error processing the prediction.");
                }
            });
        }
    });
});
