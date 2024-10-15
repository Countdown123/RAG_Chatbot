document.addEventListener("DOMContentLoaded", function () {
  var loginForm = document.getElementById("loginForm");
  var errorModal = document.getElementById("errorModal");
  var closeBtn = document.querySelector(".close");

  if (loginForm) {
    loginForm.addEventListener("submit", async function (event) {
      event.preventDefault(); // Prevent the form from submitting normally

      // Create URLSearchParams to hold form data in application/x-www-form-urlencoded format
      var formData = new URLSearchParams();
      formData.append('username', document.getElementById('email').value.trim());
      formData.append('password', document.getElementById('password').value.trim());

      try {
        // Send the login data to the backend
        let response = await fetch("/token", {
          method: "POST",
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
          body: formData.toString(),
        });

        if (response.ok) {
          // Handle successful login
          let data = await response.json();
          // Store the access token in localStorage
          localStorage.setItem("accessToken", data.access_token);
          // Redirect to the main chat page
          window.location.href = "index.html";
        } else {
          // Handle login failure
          showErrorModal();
        }
      } catch (error) {
        console.error("Error logging in:", error);
        showErrorModal();
      }
    });
  } else {
    console.error("The login form was not found.");
  }

  // Function to show the error modal
  function showErrorModal() {
    if (errorModal) {
      errorModal.style.display = "block";
    } else {
      console.error("Error modal not found");
    }
  }

  // Close the modal when the close button is clicked
  if (closeBtn) {
    closeBtn.onclick = function() {
      errorModal.style.display = "none";
    }
  }

  // Close the modal when clicking outside of it
  window.onclick = function(event) {
    if (event.target == errorModal) {
      errorModal.style.display = "none";
    }
  }
});